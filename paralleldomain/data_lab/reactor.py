import base64
import json
import logging
import random
from dataclasses import dataclass
from time import sleep
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

from paralleldomain import Dataset, Scene
from paralleldomain.constants import CAMERA_MODEL_OPENCV_FISHEYE, CAMERA_MODEL_PD_FISHEYE
from paralleldomain.data_lab.config.reactor import ReactorConfig, ReactorObject
from paralleldomain.data_lab.reactor_undistortion import distort_reactor_output, undistort_inpainting_input
from paralleldomain.decoding.in_memory.frame_decoder import InMemoryFrameDecoder
from paralleldomain.decoding.in_memory.scene_decoder import InMemorySceneDecoder
from paralleldomain.decoding.in_memory.sensor_frame_decoder import InMemoryCameraFrameDecoder
from paralleldomain.model.annotation import AnnotationTypes, InstanceSegmentation2D, SemanticSegmentation2D
from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import CameraSensorFrame, TDateTime
from paralleldomain.model.type_aliases import SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.transformation import Transformation

REACTOR_ENDPOINT: str = "https://reactor.internal.paralleldomain.com/change_shape"
REACTOR_ENDPOINT_CHANGE_TEXTURE: str = "https://reactor.internal.paralleldomain.com/change_texture"

logger = logging.getLogger(__name__)


@dataclass
class PerCameraReactorInput:
    target_image: np.ndarray
    target_frame_semseg_mask: SemanticSegmentation2D
    target_frame_instance_mask: InstanceSegmentation2D
    target_depth: Optional[np.ndarray] = None
    empty_image: Optional[np.ndarray] = None
    empty_frame_semseg_mask: Optional[SemanticSegmentation2D] = None
    empty_frame_instance_mask: Optional[InstanceSegmentation2D] = None


@dataclass
class ReactorInput:
    target_scene: Scene
    target_frame: Frame
    per_camera_input: Dict[str, PerCameraReactorInput]
    date_time: TDateTime


@dataclass
class UndistortionOutput:
    input_image: np.ndarray
    empty_image: np.ndarray
    input_mask: np.ndarray
    virtual_camera_intrinsic: np.ndarray
    virtual_camera_to_actual_sensor_in_rdf: Transformation


class ReactorInputLoader:
    def __init__(self, reactor_config: ReactorConfig, stored_dataset: Dataset = None):
        self._stored_dataset = stored_dataset
        self._reactor_config = reactor_config
        self._show_class_names = True

    def _resolve_cameras_to_use(self, sensor_names: List[SensorName]):
        ego_agent_id = sensor_names[0].split("-")[-1]
        if ego_agent_id.replace(".", "", 1).isdigit():
            return [f"{c}-{ego_agent_id}" for c in self._reactor_config.cameras_to_use]
        return self._reactor_config.cameras_to_use

    def load_reactor_input(self, input_data: Tuple[Frame, Scene]) -> ReactorInput:
        empty_frame, empty_scene = input_data
        self._reactor_config.reactor_object.set_asset_registry_class_id()
        if self._stored_dataset is not None:
            target_scene = self._stored_dataset.get_scene(scene_name=empty_scene.name)
            target_frame = target_scene.get_frame(frame_id=f"{int(empty_frame.frame_id):09d}")
            per_camera_input = dict()
            cameras_to_use = self._resolve_cameras_to_use(sensor_names=target_frame.sensor_names)
            for sensor_name in cameras_to_use:
                target_sensor_frame = target_frame.get_camera(camera_name=sensor_name)

                if sensor_name in target_frame.camera_names:
                    empty_frame_sensor_frame = empty_frame.get_camera(camera_name=sensor_name)
                    target_frame_semseg_mask, target_frame_instance_mask = get_mask_annotations(target_sensor_frame)
                    empty_frame_semseg_mask, empty_frame_instance_mask = get_mask_annotations(empty_frame_sensor_frame)

                    target_depth: Optional[np.ndarray] = None
                    if AnnotationTypes.Depth in target_sensor_frame.available_annotation_types:
                        target_depth = target_sensor_frame.get_annotations(annotation_type=AnnotationTypes.Depth).depth

                    per_camera_input[sensor_name] = PerCameraReactorInput(
                        target_image=target_sensor_frame.image.rgba,
                        target_depth=target_depth,
                        empty_image=empty_frame_sensor_frame.image.rgba,
                        target_frame_semseg_mask=target_frame_semseg_mask,
                        target_frame_instance_mask=target_frame_instance_mask,
                        empty_frame_semseg_mask=empty_frame_semseg_mask,
                        empty_frame_instance_mask=empty_frame_instance_mask,
                    )
            return ReactorInput(
                target_frame=target_frame,
                target_scene=target_scene,
                per_camera_input=per_camera_input,
                date_time=empty_frame.date_time,
            )
        else:
            raise ValueError("ReactorInput requires cached auxiliary dataset.")

    def load_reactor_input_rgbd(self, input_data: Tuple[Frame, Scene]) -> ReactorInput:
        target_frame, target_scene = input_data
        per_camera_input = dict()
        cameras_to_use = self._resolve_cameras_to_use(sensor_names=target_frame.sensor_names)
        for sensor_name in cameras_to_use:
            if sensor_name in target_frame.camera_names:
                target_sensor_frame = target_frame.get_camera(camera_name=sensor_name)
                target_frame_semseg_mask, target_frame_instance_mask = get_mask_annotations(target_sensor_frame)
                class_map = target_sensor_frame.get_class_map(annotation_type=AnnotationTypes.SemanticSegmentation2D)
                class_names = class_map.class_names
                self._reactor_config.reactor_object.set_asset_registry_class_id(class_map=class_map)
                if self._show_class_names is True:
                    logger.info("Available registry class_names are: %s", class_names)
                    # Only print this once
                    self._show_class_names = False
                if AnnotationTypes.Depth in target_sensor_frame.available_annotation_types:
                    target_depth = target_sensor_frame.get_annotations(annotation_type=AnnotationTypes.Depth).depth
                else:
                    raise ValueError("Reactor requires depth annotations.")

                per_camera_input[sensor_name] = PerCameraReactorInput(
                    target_image=target_sensor_frame.image.rgba,
                    target_depth=target_depth,
                    empty_image=None,
                    target_frame_semseg_mask=target_frame_semseg_mask,
                    target_frame_instance_mask=target_frame_instance_mask,
                    empty_frame_semseg_mask=None,
                    empty_frame_instance_mask=None,
                )
        return ReactorInput(
            target_frame=target_frame,
            target_scene=target_scene,
            per_camera_input=per_camera_input,
            date_time=target_frame.date_time,
        )


class ReactorFrameStreamGenerator:
    def __init__(self, reactor_config: ReactorConfig):
        # We assume that only get called ordered by scene, so that we don't need to keep all scenes in memory
        self._current_scene: Optional[Scene] = None
        self._current_scene_decoder: Optional[InMemorySceneDecoder] = None
        self._reactor_config = reactor_config

    def create_reactor_frame(self, reactor_input: ReactorInput) -> Tuple[Frame, Scene]:
        scene_name = reactor_input.target_scene.name
        if self._current_scene is None or self._current_scene.name != scene_name:
            self._current_scene_decoder = InMemorySceneDecoder.from_scene(reactor_input.target_scene)
            self._current_scene_decoder.camera_names = list(reactor_input.per_camera_input.keys())
            self._current_scene_decoder.frames = dict()
            self._current_scene_decoder.frame_ids = list()
            self._current_scene = Scene(
                decoder=self._current_scene_decoder,
            )

        target_frame = reactor_input.target_frame
        camera_sensor_frames = list()
        for sensor_name in reactor_input.per_camera_input.keys():
            if self._reactor_config.reactor_object.change_shape is True:
                camera_sensor_frame = self._create_changed_shape_sensor_frame(
                    reactor_input=reactor_input, sensor_name=sensor_name
                )
            else:
                camera_sensor_frame = self._create_changed_texture_sensor_frame(
                    reactor_input=reactor_input, sensor_name=sensor_name
                )
            camera_sensor_frames.append(camera_sensor_frame)

        frame_decoder = InMemoryFrameDecoder(
            frame_id=target_frame.frame_id,
            ego_pose=target_frame.ego_frame.pose,
            camera_sensor_frames=camera_sensor_frames,
            lidar_sensor_frames=list(target_frame.lidar_frames),
            radar_sensor_frames=list(target_frame.radar_frames),
            camera_names=target_frame.camera_names,
            lidar_names=target_frame.lidar_names,
            radar_names=target_frame.radar_names,
            date_time=reactor_input.date_time,
            metadata=target_frame.metadata,
            scene_name=target_frame.scene_name,
            dataset_name=target_frame.dataset_name,
        )
        final_frame = Frame(decoder=frame_decoder)
        self._current_scene_decoder.frame_ids.append(final_frame.frame_id)
        self._current_scene_decoder.frames[final_frame.frame_id] = final_frame

        return final_frame, self._current_scene

    def _create_changed_shape_sensor_frame(self, reactor_input: ReactorInput, sensor_name: str) -> CameraSensorFrame:
        target_frame = reactor_input.target_frame
        target_sensor_frame = target_frame.get_camera(camera_name=sensor_name)
        decoder = InMemoryCameraFrameDecoder.from_camera_frame(camera_frame=target_sensor_frame)

        if sensor_name in target_frame.camera_names:
            camera_reactor_input = reactor_input.per_camera_input[sensor_name]
            instance_reactor_mask, id_to_prompt = get_instance_mask_and_prompts(
                camera_reactor_input=camera_reactor_input,
                reactor_object=self._reactor_config.reactor_object,
            )
            empty_frame_semseg_mask = camera_reactor_input.empty_frame_semseg_mask
            empty_frame_instance_mask = camera_reactor_input.empty_frame_instance_mask
            camera = target_frame.get_camera(camera_name=sensor_name)

            if id_to_prompt is not None:
                undistortion_data = None
                input_image = camera_reactor_input.target_image
                empty_image = camera_reactor_input.empty_image
                input_mask = instance_reactor_mask.astype(np.uint8)
                if (
                    camera.intrinsic.camera_model in [CAMERA_MODEL_OPENCV_FISHEYE, CAMERA_MODEL_PD_FISHEYE]
                    and self._reactor_config.undistort_input
                ):
                    if reactor_input.per_camera_input[sensor_name].target_depth is None:
                        logger.warning(
                            "Can't run reactor on undistorted image without depth annotations."
                            " Falling back to distorted images."
                        )
                    elif camera.intrinsic.camera_model is CAMERA_MODEL_PD_FISHEYE and camera.distortion_lookup is None:
                        logger.warning(
                            "Can't run reactor on undistorted image without provided distortion lookup tables "
                            "in reactor_config. Falling back to distorted images."
                        )
                    else:
                        undistortion_data = undistort_inpainting_input(
                            input_image=input_image,
                            empty_image=empty_image,
                            input_mask=input_mask,
                            depth=reactor_input.per_camera_input[sensor_name].target_depth,
                            camera=camera,
                            context_scale=self._reactor_config.context_scale,
                            context_scale_pad_factor=self._reactor_config.undistort_context_scale_pad_factor,
                        )
                        input_image = undistortion_data.input_image
                        empty_image = undistortion_data.empty_image
                        input_mask = undistortion_data.input_mask
                output_image, output_mask = change_shape(
                    input_image=input_image,
                    empty_input_image=empty_image,
                    input_mask=input_mask,
                    prompt=json.dumps(id_to_prompt),
                    inference_width=self._reactor_config.inference_resolution,
                    inference_height=self._reactor_config.inference_resolution,
                    seed=self._reactor_config.random_seed,
                    noise_scheduler=self._reactor_config.noise_scheduler,
                    number_of_inference_steps=self._reactor_config.number_of_inference_steps,
                    guidance_scale=self._reactor_config.guidance_scale,
                    negative_prompt=self._reactor_config.negative_prompt,
                    use_color_matching=self._reactor_config.use_color_matching,
                    use_camera_noise=self._reactor_config.use_camera_noise,
                    clip_score_threshold=self._reactor_config.clip_score_threshold,
                    context_scale=self._reactor_config.context_scale,
                    color_matching_strength=self._reactor_config.color_matching_strength,
                )
                if np.any(output_mask):
                    if undistortion_data is not None:
                        output_image, output_mask = distort_reactor_output(
                            output_image=output_image,
                            output_mask=output_mask,
                            undistortion_data=undistortion_data,
                            input_image=camera_reactor_input.target_image,
                            camera=camera,
                            depth=reactor_input.per_camera_input[sensor_name].target_depth,
                        )

                    decoder.rgba = output_image
                    empty_frame_semseg_mask.class_ids[
                        output_mask > 0
                    ] = self._reactor_config.reactor_object.new_class_id
                    empty_frame_instance_mask.instance_ids = merge_instance_masks(
                        instance_mask=empty_frame_instance_mask.instance_ids,
                        generated_output_mask=output_mask,
                    )
                else:
                    decoder.rgba = camera_reactor_input.empty_image
            else:
                decoder.rgba = camera_reactor_input.empty_image
            for annotation in decoder.annotations:
                if annotation.annotation_type == AnnotationTypes.SemanticSegmentation2D:
                    decoder.annotations[annotation] = empty_frame_semseg_mask
                if annotation.annotation_type == AnnotationTypes.InstanceSegmentation2D:
                    decoder.annotations[annotation] = empty_frame_instance_mask
        camera_sensor_frame = CameraSensorFrame(decoder=decoder)
        return camera_sensor_frame

    def _create_changed_texture_sensor_frame(self, reactor_input: ReactorInput, sensor_name: str) -> CameraSensorFrame:
        target_frame = reactor_input.target_frame
        target_sensor_frame = target_frame.get_camera(camera_name=sensor_name)
        decoder = InMemoryCameraFrameDecoder.from_camera_frame(camera_frame=target_sensor_frame)

        if sensor_name in target_frame.camera_names:
            camera_reactor_input = reactor_input.per_camera_input[sensor_name]
            instance_mask, id_to_prompt = get_instance_mask_and_prompts(
                camera_reactor_input=camera_reactor_input,
                reactor_object=self._reactor_config.reactor_object,
            )
            target_frame_semseg_mask = camera_reactor_input.target_frame_semseg_mask
            camera = target_frame.get_camera(camera_name=sensor_name)

            if id_to_prompt is not None:
                input_image = camera_reactor_input.target_image
                depth = camera_reactor_input.target_depth
                input_mask = instance_mask.astype(np.uint8)
                if camera.intrinsic.camera_model in [CAMERA_MODEL_PD_FISHEYE, CAMERA_MODEL_OPENCV_FISHEYE]:
                    raise NotImplementedError("Reactor: can't run change texture on sensor rig with fisheye cameras.")
                output_image = change_texture(
                    input_image=input_image,
                    depth=depth,
                    input_mask=input_mask,
                    prompt=json.dumps(id_to_prompt),
                    inference_width=self._reactor_config.inference_resolution,
                    inference_height=self._reactor_config.inference_resolution,
                    seed=self._reactor_config.random_seed,
                    noise_scheduler=self._reactor_config.noise_scheduler,
                    number_of_inference_steps=self._reactor_config.number_of_inference_steps,
                    noise_strength=self._reactor_config.noise_strength,
                    guidance_scale=self._reactor_config.guidance_scale,
                    negative_prompt=self._reactor_config.negative_prompt,
                    use_color_matching=self._reactor_config.use_color_matching,
                    use_camera_noise=self._reactor_config.use_camera_noise,
                    context_scale=self._reactor_config.context_scale,
                    color_matching_strength=self._reactor_config.color_matching_strength,
                )

                if np.any(output_image):
                    decoder.rgba = output_image
                else:
                    decoder.rgba = camera_reactor_input.target_image
            else:
                decoder.rgba = camera_reactor_input.target_image
            for annotation in decoder.annotations:
                if annotation.annotation_type == AnnotationTypes.SemanticSegmentation2D:
                    decoder.annotations[annotation] = target_frame_semseg_mask
                    break
        camera_sensor_frame = CameraSensorFrame(decoder=decoder)
        return camera_sensor_frame


def get_instance_mask_and_prompts(
    camera_reactor_input: PerCameraReactorInput,
    reactor_object: ReactorObject,
):
    if reactor_object.change_shape is False:
        instance_mask, id_to_prompt = get_instance_masks_and_prompts_from_class_id(
            camera_reactor_input=camera_reactor_input, reactor_object=reactor_object
        )
        return instance_mask, id_to_prompt
    else:
        instance_mask, id_to_prompt = get_instance_mask_and_prompts_from_paired_frame(
            camera_reactor_input=camera_reactor_input, reactor_object=reactor_object
        )
        return instance_mask, id_to_prompt


def get_instance_masks_and_prompts_from_class_id(
    camera_reactor_input: PerCameraReactorInput, reactor_object: ReactorObject
):
    orig_mask = np.copy(camera_reactor_input.target_frame_instance_mask.instance_ids)
    instance_mask = np.zeros_like(orig_mask)
    semseg_mask = camera_reactor_input.target_frame_semseg_mask.class_ids
    orig_mask[semseg_mask != reactor_object.registry_class_id] = 0
    # Offset instance_ids to start with id = 0
    instance_ids = np.unique(orig_mask)
    if len(instance_ids) > 254:
        raise ValueError(f"Reactor: marked object instances {len(instance_ids)} exceed limit of 254 per frame.")
    instance_id_offset = 0
    if instance_ids[0] != 0:
        instance_id_offset = 1
    for new_id, instance_id in enumerate(instance_ids):
        instance_mask[orig_mask == instance_id] = new_id + instance_id_offset

    id_to_prompt = dict()
    instance_ids = np.unique(instance_mask)
    for instance_id in instance_ids:
        if instance_id == 0:
            continue
        if random.random() < reactor_object.replacement_probability:
            id_to_prompt[int(instance_id)] = random.choice(reactor_object.prompts)
        else:
            instance_mask[instance_mask == instance_id] = 0
    return instance_mask, id_to_prompt


def get_instance_mask_and_prompts_from_paired_frame(
    camera_reactor_input: PerCameraReactorInput,
    reactor_object: ReactorObject,
):
    instance_mask = np.copy(camera_reactor_input.target_frame_instance_mask.instance_ids)
    semseg_mask = camera_reactor_input.target_frame_semseg_mask.class_ids
    empty_frame_instance_mask = np.copy(camera_reactor_input.empty_frame_instance_mask.instance_ids)
    empty_frame_semseg_mask = camera_reactor_input.empty_frame_semseg_mask.class_ids

    instance_mask[semseg_mask != reactor_object.registry_class_id] = 0
    empty_frame_instance_mask[empty_frame_semseg_mask != reactor_object.registry_class_id] = 0

    target_mask = np.zeros_like(instance_mask)
    num_instance_ids = len(np.unique(instance_mask))
    empty_frame_num_instance_ids = len(np.unique(empty_frame_instance_mask))
    if reactor_object.registry_class_id not in semseg_mask:
        logger.warning(
            f"Reactor: the frame does not contain a semseg mask that matches the class_id "
            f"{reactor_object.registry_class_id} of the proxy object {reactor_object.asset_name}. "
            "Will skip this frame"
        )
        return target_mask, None
    if num_instance_ids == empty_frame_num_instance_ids:
        logger.warning(
            f"Reactor: no instance mask found for the proxy object {reactor_object.asset_name}. "
            f"Will skip this frame."
        )
        # Nothing changed, no target mask found
        return target_mask, None
    else:
        # take instance mask with the largest pixel difference and count as target mask
        diff_mask = np.absolute(instance_mask - empty_frame_instance_mask)
        average_diff = np.average(np.unique(diff_mask))
        diff_mask = diff_mask > average_diff
        instance_ids_changed_pixels_count = np.bincount(instance_mask[diff_mask])
        instance_id = np.argmax(instance_ids_changed_pixels_count)
        id_to_prompt = dict()
        target_mask[instance_mask == instance_id] = 1
        id_to_prompt[1] = random.choice(reactor_object.prompts)
    return target_mask, id_to_prompt


def merge_instance_masks(instance_mask: np.ndarray, generated_output_mask: np.ndarray) -> np.ndarray:
    output_instance_mask = np.copy(instance_mask)
    # first we set the instance_mask to zero, where we will place the generated object
    output_instance_mask[generated_output_mask > 0] = 0
    generated_output_mask = generated_output_mask.astype(instance_mask.dtype)
    # then we paste the inpainted object and assign new instance id
    generated_output_mask[generated_output_mask > 0] += np.max(instance_mask)
    output_instance_mask += generated_output_mask
    return output_instance_mask


def get_mask_annotations(
    camera_frame: CameraSensorFrame[TDateTime],
) -> Tuple[SemanticSegmentation2D, InstanceSegmentation2D]:
    if SemanticSegmentation2D not in camera_frame.available_annotation_types:
        raise ValueError("Reactor requires annotation type SemanticSegmentation2D")
    if InstanceSegmentation2D not in camera_frame.available_annotation_types:
        raise ValueError("Reactor requires annotation type InstanceSegmentation2D")
    instance_mask = camera_frame.get_annotations(annotation_type=InstanceSegmentation2D)
    semseg_mask = camera_frame.get_annotations(annotation_type=SemanticSegmentation2D)
    return semseg_mask, instance_mask


def encode_rgba_image_png_base64(image: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".png", image)[1]).decode("ascii")


def decode_rgba_image_png_base64(image_str: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(image_str), dtype=np.uint8), flags=cv2.IMREAD_UNCHANGED)


def encode_depth_base64(depth: np.ndarray) -> str:
    if np.isnan(depth).any():
        raise ValueError("Can not encode depth map that contains NaN values.")
    return base64.b64encode(depth).decode("ascii")


def invert_depth(depth: np.ndarray, epsilon=0.0001) -> np.ndarray:
    output = np.copy(depth)
    return (1 / (output + epsilon)).astype(np.float32)


def check_response_json(data: dict):
    valid_response = True
    if "output_image" not in data:
        valid_response = False
    if "output_mask" not in data:
        valid_response = False
    if valid_response is False:
        raise ValueError("The reactor endpoint did not return valid data.")


def change_shape(
    prompt: str,
    inference_width: int,
    inference_height: int,
    input_image: np.ndarray = None,
    empty_input_image: np.ndarray = None,
    input_mask: np.ndarray = None,
    negative_prompt: str = "",
    number_of_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    noise_scheduler: str = "uni_pc",
    ddim_eta: int = 0,
    seed: int = 42,
    use_color_matching: bool = True,
    use_camera_noise: bool = False,
    clip_score_threshold: float = 22,
    f1_score_threshold: float = 0.5,
    context_scale: float = 2.0,
    color_matching_strength: float = 0.5,
    maximum_number_retries: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    if input_image is None:
        raise ValueError("Input image can not be None")
    if input_mask is None:
        raise ValueError("Input instance mask can not be None.")
    if empty_input_image is None:
        raise ValueError("Empty input image can not be None")
    image_encode = encode_rgba_image_png_base64(image=input_image)
    empty_image_encode = encode_rgba_image_png_base64(image=empty_input_image)
    image_mask_encode = encode_rgba_image_png_base64(image=input_mask.squeeze())
    payload = {
        "image": image_encode,
        "empty_image": empty_image_encode,
        "image_mask": image_mask_encode,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "inference_width": inference_width,
        "inference_height": inference_height,
        "guidance_scale": guidance_scale,
        "num_diffusion_steps": number_of_inference_steps,
        "scheduler": noise_scheduler,
        "eta": ddim_eta,
        "seed": seed,
        "use_hue_transformer": use_color_matching,
        "use_camera_noise": use_camera_noise,
        "clip_score_threshold": clip_score_threshold,
        "f1_score_threshold": f1_score_threshold,
        "context_scale": context_scale,
        "hue_transform_strength": color_matching_strength,
        "maximum_number_retries": maximum_number_retries,
    }

    resp = post_request_with_retries(url=REACTOR_ENDPOINT, payload=payload)
    resp_json = resp.json()
    check_response_json(resp_json)
    output_image = decode_rgba_image_png_base64(resp_json["output_image"])
    output_mask = decode_rgba_image_png_base64(resp_json["output_mask"])
    output_mask = np.expand_dims(output_mask, 2)
    return output_image, output_mask


def change_texture(
    prompt: str,
    inference_width: int,
    inference_height: int,
    input_image: np.ndarray = None,
    depth: np.ndarray = None,
    input_mask: np.ndarray = None,
    negative_prompt: str = "",
    number_of_inference_steps: int = 20,
    noise_strength: float = 0.5,
    guidance_scale: float = 7.5,
    noise_scheduler: str = "dpm++",
    ddim_eta: int = 0,
    seed: int = 42,
    use_color_matching: bool = True,
    use_camera_noise: bool = False,
    context_scale: float = 2.0,
    color_matching_strength: float = 0.5,
) -> np.ndarray:
    if input_image is None:
        raise ValueError("Input image can not be None")
    if input_mask is None:
        raise ValueError("Input instance mask can not be None.")
    if depth is None:
        raise ValueError("Depth map can not be None")
    image_encode = encode_rgba_image_png_base64(image=input_image)
    inverse_depth = invert_depth(depth)
    depth_encode = encode_depth_base64(depth=inverse_depth)
    image_mask_encode = encode_rgba_image_png_base64(image=input_mask.squeeze())
    payload = {
        "image": image_encode,
        "image_mask": image_mask_encode,
        "depth": depth_encode,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "inference_width": inference_width,
        "inference_height": inference_height,
        "guidance_scale": guidance_scale,
        "num_diffusion_steps": number_of_inference_steps,
        "noise_strength": noise_strength,
        "scheduler": noise_scheduler,
        "eta": ddim_eta,
        "seed": seed,
        "use_hue_transformer": use_color_matching,
        "use_camera_noise": use_camera_noise,
        "context_scale": context_scale,
        "hue_transform_strength": color_matching_strength,
    }

    resp = post_request_with_retries(url=REACTOR_ENDPOINT_CHANGE_TEXTURE, payload=payload)
    resp_json = resp.json()
    output_image = decode_rgba_image_png_base64(resp_json["output_image"])
    return output_image


def clear_output_path(output_path: AnyPath, scene_indices: List[int]):
    for scene_index in scene_indices:
        scene = f"scene_{scene_index:06d}"
        for scene_path in output_path.glob(pattern=scene):
            if scene_path.is_cloud_path is True:
                scene_json_paths = [j for j in scene_path.glob(pattern="scene*.json")]
                if len(scene_json_paths) > 0:
                    raise ValueError(
                        f"Path to scene is not empty {scene_json_paths} and deletion of files in s3 is not supported. "
                        f"To use cached_reactor_states, manually delete scenes in {output_path} "
                        f"before starting the script."
                    )
            [i.rm(missing_ok=True) for i in scene_path.rglob(pattern="*.png")]
            [j.rm(missing_ok=True) for j in scene_path.rglob(pattern="*.json")]
            if scene_path.is_file():
                scene_path.rm(missing_ok=True)


def post_request_with_retries(
    url: str, payload: dict, num_retries: int = 3, sleep_seconds: float = 5
) -> requests.Response:
    response = None
    for _ in range(num_retries):
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response
        sleep(sleep_seconds)
    response.raise_for_status()
