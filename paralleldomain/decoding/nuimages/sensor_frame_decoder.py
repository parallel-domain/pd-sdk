import base64
import logging
from datetime import datetime
from typing import Any, ByteString, Dict, List, Optional, Tuple, Type, TypeVar, Union

import cv2
import numpy as np

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.nuimages.common import NUIMAGES_IMU_TO_INTERNAL_CS, NuImagesDataAccessMixin
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, TDateTime
from paralleldomain.model.annotation import (
    Annotation,
    AnnotationType,
    AnnotationTypes,
    BoundingBox2D,
    BoundingBoxes2D,
    InstanceSegmentation2D,
    SemanticSegmentation2D,
)
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.image import Image
from paralleldomain.model.point_cloud import PointCloud
from paralleldomain.model.sensor import SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image
from paralleldomain.utilities.transformation import Transformation

T = TypeVar("T")
F = TypeVar("F", Image, PointCloud, Annotation)
logger = logging.getLogger(__name__)


class NuImagesCameraSensorFrameDecoder(CameraSensorFrameDecoder[datetime], NuImagesDataAccessMixin):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        split_name: str,
        scene_name: SceneName,
        settings: DecoderSettings,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        CameraSensorFrameDecoder.__init__(
            self=self, dataset_name=dataset_name, scene_name=scene_name, settings=settings
        )
        NuImagesDataAccessMixin.__init__(
            self=self, dataset_name=dataset_name, split_name=split_name, dataset_path=self._dataset_path
        )

    def _decode_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorIntrinsic:
        sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
            log_token=self.scene_name, frame_id=frame_id, sensor_name=sensor_name
        )
        data = self.nu_samples_data[sample_data_id]
        calib_sensor_token = data["calibrated_sensor_token"]
        calib_sensor = self.nu_calibrated_sensors[calib_sensor_token]
        distortion_params = [
            calib_sensor["camera_distortion"][i] if i < len(calib_sensor["camera_distortion"]) else 0.0
            for i in range(8)
        ]
        return SensorIntrinsic(
            fx=calib_sensor["camera_intrinsic"][0][0],
            fy=calib_sensor["camera_intrinsic"][1][1],
            cx=calib_sensor["camera_intrinsic"][0][2],
            cy=calib_sensor["camera_intrinsic"][1][2],
            k1=distortion_params[0],
            k2=distortion_params[1],
            p1=distortion_params[2],
            p2=distortion_params[3],
            k3=distortion_params[4],
            k4=distortion_params[5],
            k5=distortion_params[6],
            k6=distortion_params[7],
        )

    def _decode_image_dimensions(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
            log_token=self.scene_name, frame_id=frame_id, sensor_name=sensor_name
        )
        data = self.nu_samples_data[sample_data_id]
        return int(data["height"]), int(data["width"]), 3

    def _decode_image_rgba(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
            log_token=self.scene_name, frame_id=frame_id, sensor_name=sensor_name
        )
        data = self.nu_samples_data[sample_data_id]

        img_path = AnyPath(self._dataset_path) / data["filename"]
        image_data = read_image(path=img_path, convert_to_rgb=True)

        ones = np.ones((*image_data.shape[:2], 1), dtype=image_data.dtype) * 255
        concatenated = np.concatenate([image_data, ones], axis=-1)
        return concatenated

    def _decode_available_annotation_types(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> Dict[AnnotationType, AnnotationIdentifier]:
        anno_types = dict()
        if self.split_name != "v1.0-test":
            sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
                log_token=self.scene_name, frame_id=frame_id, sensor_name=sensor_name
            )
            if sample_data_id in self.nu_sample_data_tokens_to_available_anno_types:
                has_surface, has_obj = self.nu_sample_data_tokens_to_available_anno_types[sample_data_id]
                if has_surface:
                    anno_types[AnnotationTypes.SemanticSegmentation2D] = "SemanticSegmentation2D"
                if has_obj:
                    anno_types[AnnotationTypes.InstanceSegmentation2D] = "InstanceSegmentation2D"
                    anno_types[AnnotationTypes.BoundingBoxes2D] = "BoundingBoxes2D"
        return anno_types

    def _decode_metadata(self, sensor_name: SensorName, frame_id: FrameId) -> Dict[str, Any]:
        return {}

    def _decode_class_maps(self) -> Dict[AnnotationType, ClassMap]:
        return self.nu_class_maps

    def _decode_date_time(self, sensor_name: SensorName, frame_id: FrameId) -> datetime:
        return datetime.fromtimestamp(int(frame_id) / 1000000)

    def _decode_extrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorExtrinsic:
        sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
            log_token=self.scene_name, frame_id=frame_id, sensor_name=sensor_name
        )
        data = self.nu_samples_data[sample_data_id]
        calib_sensor_token = data["calibrated_sensor_token"]
        calib_sensor = self.nu_calibrated_sensors[calib_sensor_token]
        translation = np.asarray(calib_sensor["translation"]).reshape(-1)
        if len(translation) != 3:
            logger.warning(f"Got a translation with invalid shape {translation.shape}!")
            translation = translation[:3]
        trans = Transformation(quaternion=calib_sensor["rotation"], translation=translation)
        trans = NUIMAGES_IMU_TO_INTERNAL_CS @ trans

        return trans

    def _decode_sensor_pose(self, sensor_name: SensorName, frame_id: FrameId) -> SensorPose:
        sensor_to_ego = self.get_extrinsic(sensor_name=sensor_name, frame_id=frame_id)
        ego_to_world = self.get_ego_pose(log_token=self.scene_name, frame_id=frame_id)
        sensor_to_world = ego_to_world @ sensor_to_ego.transformation_matrix
        return SensorPose.from_transformation_matrix(sensor_to_world)

    def _decode_file_path(self, sensor_name: SensorName, frame_id: FrameId, data_type: Type[F]) -> Optional[AnyPath]:
        if issubclass(data_type, Image):
            sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
                log_token=self.scene_name, frame_id=frame_id, sensor_name=sensor_name
            )
            if sample_data_id is not None:
                data = self.nu_samples_data[sample_data_id]

                img_path = AnyPath(self._dataset_path) / data["filename"]
                return img_path
        return None

    def _decode_annotations(
        self, sensor_name: SensorName, frame_id: FrameId, identifier: AnnotationIdentifier, annotation_type: T
    ) -> T:
        if issubclass(annotation_type, SemanticSegmentation2D):
            class_ids = self._decode_semantic_segmentation_2d(sensor_name=sensor_name, frame_id=frame_id)
            return SemanticSegmentation2D(class_ids=class_ids)
        elif issubclass(annotation_type, InstanceSegmentation2D):
            instance_ids = self._decode_instance_segmentation_2d(sensor_name=sensor_name, frame_id=frame_id)
            return InstanceSegmentation2D(instance_ids=instance_ids)
        elif issubclass(annotation_type, BoundingBoxes2D):
            boxes = self._decode_bounding_boxes_2d(sensor_name=sensor_name, frame_id=frame_id)
            return BoundingBoxes2D(boxes=boxes)
        else:
            raise NotImplementedError(f"{annotation_type} is not supported!")

    def _decode_semantic_segmentation_2d(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        semseg_mask = np.zeros((900, 1600)).astype(int)

        sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
            log_token=self.scene_name, frame_id=frame_id, sensor_name=sensor_name
        )
        for surface_annotation in self.nu_surface_ann[sample_data_id]:
            mask = mask_decode(surface_annotation["mask"])
            category_token = surface_annotation["category_token"]
            category_name = self.nu_category[category_token]["name"]

            # Draw mask for semantic segmentation.
            semseg_mask[mask == 1] = self.nu_name_to_index[category_name]

        if sample_data_id in self.nu_object_ann:
            object_anns = self.nu_object_ann[sample_data_id]
            object_anns = sorted(object_anns, key=lambda k: k["token"])

            for ann in object_anns:
                if ann["mask"] is None:
                    continue
                # Get color, box, mask and name.
                category_token = ann["category_token"]
                category_name = self.nu_category[category_token]["name"]

                mask = mask_decode(ann["mask"])
                # Draw masks for semantic segmentation and instance segmentation.
                semseg_mask[mask == 1] = self.nu_name_to_index[category_name]

        return np.expand_dims(semseg_mask, axis=-1)

    def _decode_instance_segmentation_2d(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        instanceseg_mask = np.zeros((900, 1600)).astype(int)

        sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
            log_token=self.scene_name, frame_id=frame_id, sensor_name=sensor_name
        )

        object_anns = self.nu_object_ann[sample_data_id]
        object_anns = sorted(object_anns, key=lambda k: k["token"])

        for i, ann in enumerate(object_anns, start=1):
            if ann["mask"] is None:
                continue
            mask = mask_decode(ann["mask"])
            instanceseg_mask[mask == 1] = i

        return np.expand_dims(instanceseg_mask, axis=-1)

    def _decode_bounding_boxes_2d(self, sensor_name: SensorName, frame_id: FrameId) -> List[BoundingBox2D]:
        sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
            log_token=self.scene_name, frame_id=frame_id, sensor_name=sensor_name
        )

        boxes = list()
        for i, ann in enumerate(self.nu_object_ann[sample_data_id], start=1):
            box = ann["bbox"]
            category_token = ann["category_token"]
            attribute_tokens = ann["attribute_tokens"]
            category_name = self.nu_category[category_token]["name"]
            attributes = {self.nu_attribute[tk]["name"]: self.nu_attribute[tk] for tk in attribute_tokens}
            class_id = self.nu_name_to_index[category_name]
            boxes.append(
                BoundingBox2D(
                    x=box[0],
                    y=box[1],
                    width=box[2] - box[0],
                    height=box[3] - box[1],
                    class_id=class_id,
                    instance_id=i,
                    attributes=attributes,
                )
            )
        return boxes


def mask_decode(mask: dict) -> np.ndarray:
    """
    Decode the mask from base64 string to binary string, then decoded to get a mask.
    :param mask: The mask dictionary with fields `size` and `counts`.
    :return: A numpy array representing the binary mask for this class.
    """
    new_mask = mask.copy()
    new_mask["counts"] = base64.b64decode(mask["counts"])
    mask_rle = _rle_from_leb_string(leb_bytes=new_mask["counts"])
    return _rle_to_mask(mask_rle=mask_rle, shape=new_mask["size"])


def _rle_to_mask(mask_rle: List[int], shape: Tuple[int, int]) -> np.ndarray:
    flat_size = shape[0] * shape[1]
    mask = np.zeros(flat_size, dtype=int)
    rle_np = np.array([a for a in mask_rle], dtype=int)
    rle_np_cum = np.cumsum(rle_np)
    one_slice = np.array(list(zip(rle_np_cum[::2], rle_np_cum[1::2])))
    for start, stop in one_slice:
        mask[start:stop] = 1
    return mask.reshape(shape, order="F")


def _rle_from_leb_string(leb_bytes: ByteString) -> List[int]:
    """
    Coco custom LEB128 decoding. See https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c#L205
    """
    m = 0
    p = 0
    cnts = list()
    max_st_len = len(leb_bytes)
    while p < max_st_len:
        x = 0
        k = 0
        more = True
        while more:
            c = leb_bytes[p] - 48
            x |= (c & 0x1F) << 5 * k
            more = c & 0x20 == 32
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << 5 * k
        if m > 2:
            x += cnts[m - 2]
        cnts.append(x)
        m += 1
    return cnts
