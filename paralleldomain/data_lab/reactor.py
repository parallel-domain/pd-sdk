import base64
import json
import logging
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import cv2
import numpy as np
import requests
from paralleldomain import Scene, Dataset
from paralleldomain.constants import CAMERA_MODEL_OPENCV_FISHEYE, CAMERA_MODEL_PD_FISHEYE
from paralleldomain.data_lab.config.reactor import ReactorObject, ReactorConfig
from paralleldomain.data_lab.reactor_undistortion import undistort_inpainting_input, distort_reactor_output
from paralleldomain.decoding.in_memory.frame_decoder import InMemoryFrameDecoder
from paralleldomain.decoding.in_memory.scene_decoder import InMemorySceneDecoder
from paralleldomain.decoding.in_memory.sensor_frame_decoder import InMemoryCameraFrameDecoder
from paralleldomain.model.annotation import (
    InstanceSegmentation2D,
    SemanticSegmentation2D,
    AnnotationTypes,
    AnnotationIdentifier,
)
from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import CameraSensorFrame, TDateTime
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.transformation import Transformation

REACTOR_ENDPOINT: str = "https://reactor.internal.paralleldomain.com/change_shape"

logger = logging.getLogger(__name__)


@dataclass()
class PerCameraReactorInput:
    paired_image: np.ndarray
    paired_depth: Optional[np.ndarray]
    empty_image: np.ndarray
    paired_frame_semseg_mask: SemanticSegmentation2D
    paired_frame_instance_mask: InstanceSegmentation2D
    empty_frame_semseg_mask: SemanticSegmentation2D
    empty_frame_instance_mask: InstanceSegmentation2D


@dataclass()
class ReactorInput:
    paired_scene: Scene
    paired_frame: Frame
    per_camera_input: Dict[str, PerCameraReactorInput]


@dataclass()
class UndistortionOutput:
    input_image: np.ndarray
    empty_image: np.ndarray
    input_mask: np.ndarray
    virtual_camera_intrinsic: np.ndarray
    virtual_camera_to_actual_sensor_in_rdf: Transformation


class ReactorInputLoader:
    def __init__(self, stored_dataset: Dataset, reactor_config: ReactorConfig):
        self._stored_dataset = stored_dataset
        self._reactor_config = reactor_config

    def load_reactor_input(self, input_data: Tuple[Frame, Scene]) -> ReactorInput:
        empty_frame, empty_scene = input_data
        paired_scene = self._stored_dataset.get_scene(scene_name=empty_scene.name)
        paired_frame = paired_scene.get_frame(frame_id=empty_frame.frame_id)

        per_camera_input = dict()
        for sensor_name in self._reactor_config.cameras_to_use:
            paired_sensor_frame = paired_frame.get_camera(camera_name=sensor_name)

            if sensor_name in paired_frame.camera_names:
                empty_frame_sensor_frame = empty_frame.get_camera(camera_name=sensor_name)
                paired_frame_semseg_mask, paired_frame_instance_mask = get_mask_annotations(paired_sensor_frame)
                empty_frame_semseg_mask, empty_frame_instance_mask = get_mask_annotations(empty_frame_sensor_frame)

                paired_depth: Optional[np.ndarray] = None
                if AnnotationTypes.Depth in paired_sensor_frame.available_annotation_types:
                    paired_depth = paired_sensor_frame.get_annotations(annotation_type=AnnotationTypes.Depth).depth

                per_camera_input[sensor_name] = PerCameraReactorInput(
                    paired_image=paired_sensor_frame.image.rgba,
                    paired_depth=paired_depth,
                    empty_image=empty_frame_sensor_frame.image.rgba,
                    paired_frame_semseg_mask=paired_frame_semseg_mask,
                    paired_frame_instance_mask=paired_frame_instance_mask,
                    empty_frame_semseg_mask=empty_frame_semseg_mask,
                    empty_frame_instance_mask=empty_frame_instance_mask,
                )
        return ReactorInput(paired_frame=paired_frame, paired_scene=paired_scene, per_camera_input=per_camera_input)


class ReactorFrameStreamGenerator:
    def __init__(self, reactor_config: ReactorConfig):
        # We assume that only get called ordered by scene, so that we don't need to keep all scenes in memory
        self._current_scene: Optional[Scene] = None
        self._current_scene_decoder: Optional[InMemorySceneDecoder] = None
        self._reactor_config = reactor_config

    def create_reactor_frame(self, reactor_input: ReactorInput) -> Tuple[Frame, Scene]:
        scene_name = reactor_input.paired_scene.name
        if self._current_scene is None or self._current_scene.name != scene_name:
            self._current_scene_decoder = InMemorySceneDecoder.from_scene(reactor_input.paired_scene)
            self._current_scene_decoder.camera_names = self._reactor_config.cameras_to_use
            self._current_scene_decoder.frames = dict()
            self._current_scene_decoder.frame_ids = list()
            self._current_scene = Scene(
                decoder=self._current_scene_decoder,
                name=scene_name,
            )

        paired_frame = reactor_input.paired_frame
        camera_sensor_frames = list()
        for sensor_name in self._reactor_config.cameras_to_use:
            camera_sensor_frame = self._create_reactor_sensor_frame(
                reactor_input=reactor_input, sensor_name=sensor_name
            )
            camera_sensor_frames.append(camera_sensor_frame)

        frame_decoder = InMemoryFrameDecoder(
            ego_pose=paired_frame.ego_frame.pose,
            camera_sensor_frames=camera_sensor_frames,
            lidar_sensor_frames=list(paired_frame.lidar_frames),
            radar_sensor_frames=list(paired_frame.radar_frames),
            camera_names=paired_frame.camera_names,
            lidar_names=paired_frame.lidar_names,
            radar_names=paired_frame.radar_names,
            date_time=paired_frame.date_time,
            metadata=paired_frame.metadata,
            scene_name=paired_frame.scene_name,
            dataset_name=paired_frame.dataset_name,
        )
        final_frame = Frame(
            frame_id=paired_frame.frame_id,
            decoder=frame_decoder,
        )
        self._current_scene_decoder.frame_ids.append(final_frame.frame_id)
        self._current_scene_decoder.frames[final_frame.frame_id] = final_frame

        return final_frame, self._current_scene

    def _create_reactor_sensor_frame(self, reactor_input: ReactorInput, sensor_name: str) -> CameraSensorFrame:
        paired_frame = reactor_input.paired_frame
        paired_sensor_frame = paired_frame.get_camera(camera_name=sensor_name)
        decoder = InMemoryCameraFrameDecoder.from_camera_frame(camera_frame=paired_sensor_frame)

        if sensor_name in paired_frame.camera_names:
            camera_inpainting_input = reactor_input.per_camera_input[sensor_name]
            instance_inpainting_mask, id_to_prompt = get_instance_mask_and_prompts(
                camera_inpainting_input=camera_inpainting_input,
                inpainting_object=self._reactor_config.reactor_object,
            )
            empty_frame_semseg_mask = camera_inpainting_input.empty_frame_semseg_mask
            empty_frame_instance_mask = camera_inpainting_input.empty_frame_instance_mask
            camera = paired_frame.get_camera(camera_name=sensor_name)

            if id_to_prompt is not None:
                undistortion_data = None
                input_image = camera_inpainting_input.paired_image
                empty_image = camera_inpainting_input.empty_image
                input_mask = instance_inpainting_mask.astype(np.uint8)
                # TODO: we currently don't have access to the PD_FISHEYE distortion lookup here, so we can't undistort
                if (
                    camera.intrinsic.camera_model in [CAMERA_MODEL_OPENCV_FISHEYE]
                    and self._reactor_config.undistort_input
                ):
                    if reactor_input.per_camera_input[sensor_name].paired_depth is None:
                        logger.warning(
                            "Can't run reactor on undistorted image without depth annotations."
                            " Falling back to distorted images."
                        )
                    else:
                        undistortion_data = undistort_inpainting_input(
                            input_image=input_image,
                            empty_image=empty_image,
                            input_mask=input_mask,
                            depth=reactor_input.per_camera_input[sensor_name].paired_depth,
                            camera=camera,
                            context_scale=self._reactor_config.context_scale,
                            context_scale_pad_factor=self._reactor_config.undistort_context_scale_pad_factor,
                        )
                        input_image = undistortion_data.input_image
                        empty_image = undistortion_data.empty_image
                        input_mask = undistortion_data.input_mask
                if camera.intrinsic.camera_model in [CAMERA_MODEL_PD_FISHEYE] and self._reactor_config.undistort_input:
                    logger.warning(
                        "Can't run reactor on undistorted image without depth annotations."
                        " Falling back to distorted images."
                    )
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
                            input_image=camera_inpainting_input.paired_image,
                            camera=camera,
                            depth=reactor_input.per_camera_input[sensor_name].paired_depth,
                        )

                    decoder.rgba = output_image
                    empty_frame_semseg_mask.class_ids[
                        output_mask > 0
                    ] = self._reactor_config.reactor_object.new_class_id
                    empty_frame_instance_mask.instance_ids = merge_instance_masks(
                        instance_mask=empty_frame_instance_mask.instance_ids,
                        inpainting_output_mask=output_mask,
                    )
                else:
                    decoder.rgba = camera_inpainting_input.empty_image
            else:
                decoder.rgba = camera_inpainting_input.empty_image
            decoder.annotations[AnnotationIdentifier(AnnotationTypes.SemanticSegmentation2D)] = empty_frame_semseg_mask
            decoder.annotations[
                AnnotationIdentifier(AnnotationTypes.InstanceSegmentation2D)
            ] = empty_frame_instance_mask
        camera_sensor_frame = CameraSensorFrame(
            sensor_name=paired_sensor_frame.sensor_name,
            frame_id=paired_sensor_frame.frame_id,
            decoder=decoder,
        )
        return camera_sensor_frame


def get_instance_mask_and_prompts(
    camera_inpainting_input: PerCameraReactorInput,
    inpainting_object: ReactorObject,
):
    instance_mask = np.copy(camera_inpainting_input.paired_frame_instance_mask.instance_ids)
    semseg_mask = camera_inpainting_input.paired_frame_semseg_mask.class_ids
    empty_frame_instance_mask = np.copy(camera_inpainting_input.empty_frame_instance_mask.instance_ids)
    empty_frame_semseg_mask = camera_inpainting_input.empty_frame_semseg_mask.class_ids

    instance_mask[semseg_mask != inpainting_object.registry_class_id] = 0
    empty_frame_instance_mask[empty_frame_semseg_mask != inpainting_object.registry_class_id] = 0

    inpainting_mask = np.zeros_like(instance_mask)
    num_instance_ids = len(np.unique(instance_mask))
    empty_frame_num_instance_ids = len(np.unique(empty_frame_instance_mask))
    if inpainting_object.registry_class_id not in semseg_mask:
        logger.warning(
            f"Reactor: the frame does not contain a semseg mask that matches the class_id "
            f"{inpainting_object.registry_class_id} of the proxy object {inpainting_object.asset_name}. "
            "Will skip this frame"
        )
        return inpainting_mask, None
    if num_instance_ids == empty_frame_num_instance_ids:
        logger.warning(
            f"Reactor: no instance mask found for the proxy object {inpainting_object.asset_name}. "
            f"Will skip this frame."
        )
        # Nothing changed, no inpainting mask found
        return inpainting_mask, None
    else:
        # take instance mask with the largest pixel difference and count as inpainting mask
        diff_mask = np.absolute(instance_mask - empty_frame_instance_mask)
        average_diff = np.average(np.unique(diff_mask))
        diff_mask = diff_mask > average_diff
        instance_ids_changed_pixels_count = np.bincount(instance_mask[diff_mask])
        instance_id = np.argmax(instance_ids_changed_pixels_count)
        id_to_prompt = dict()
        inpainting_mask[instance_mask == instance_id] = 1
        id_to_prompt[1] = random.choice(inpainting_object.prompts)
    return inpainting_mask, id_to_prompt


def merge_instance_masks(instance_mask: np.ndarray, inpainting_output_mask: np.ndarray) -> np.ndarray:
    output_instance_mask = np.copy(instance_mask)
    # first we set the instance_mask to zero, where we will place the inpainted object
    output_instance_mask[inpainting_output_mask > 0] = 0
    inpainting_output_mask = inpainting_output_mask.astype(instance_mask.dtype)
    # then we paste the inpainted object and assign new instance id
    inpainting_output_mask[inpainting_output_mask > 0] += np.max(instance_mask)
    output_instance_mask += inpainting_output_mask
    return output_instance_mask


def get_mask_annotations(
    camera_frame: CameraSensorFrame[TDateTime],
) -> Tuple[SemanticSegmentation2D, InstanceSegmentation2D]:
    if SemanticSegmentation2D not in camera_frame.available_annotation_types:
        raise ValueError("Instance inpainting requires annotation type SemanticSegmentation2D")
    if InstanceSegmentation2D not in camera_frame.available_annotation_types:
        raise ValueError("Instance inpainting requires annotation type InstanceSegmentation2D")
    instance_mask = camera_frame.get_annotations(annotation_type=InstanceSegmentation2D)
    semseg_mask = camera_frame.get_annotations(annotation_type=SemanticSegmentation2D)
    return semseg_mask, instance_mask


def encode_rgba_image_png_base64(image: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".png", image)[1]).decode("ascii")


def decode_rgba_image_png_base64(image_str: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(image_str), dtype=np.uint8), flags=cv2.IMREAD_UNCHANGED)


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

    resp = requests.post(REACTOR_ENDPOINT, json=payload)
    resp.raise_for_status()
    resp_json = resp.json()
    check_response_json(resp_json)
    output_image = decode_rgba_image_png_base64(resp_json["output_image"])
    output_mask = decode_rgba_image_png_base64(resp_json["output_mask"])
    output_mask = np.expand_dims(output_mask, 2)
    return output_image, output_mask


def clear_output_path(output_path: AnyPath):
    scene_paths = [a for a in output_path.glob(pattern="scene_*")]
    for scene in scene_paths:
        [i.rm(missing_ok=True) for i in scene.rglob(pattern="*.png")]
        [j.rm(missing_ok=True) for j in scene.rglob(pattern="*.json")]
        [p.rm(missing_ok=True) for p in scene.rglob(pattern="*.pickle")]
        if scene.is_file():
            scene.rm(missing_ok=True)
