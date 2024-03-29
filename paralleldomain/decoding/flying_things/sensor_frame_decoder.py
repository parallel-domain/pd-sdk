import re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import imagesize
import numpy as np

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.flying_things.common import (
    OPTICAL_FLOW_BACKWARD_DIRECTION_NAME,
    OPTICAL_FLOW_FORWARD_DIRECTION_NAME,
    frame_id_to_timestamp,
    get_image_folder_name,
    get_scene_flow_folder,
    get_scene_folder,
)
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationTypes, BackwardOpticalFlow, OpticalFlow
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import SensorDataCopyTypes, SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_flo, read_image

T = TypeVar("T")


class FlyingThingsCameraSensorFrameDecoder(CameraSensorFrameDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        frame_id: FrameId,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        split_name: str,
        split_list: List[int],
        is_driving_subset: bool,
        is_unordered_scene: bool,
        scene_decoder,
        is_full_dataset_format: bool = False,
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        self._is_driving_subset = is_driving_subset
        self._is_full_dataset_format = is_full_dataset_format
        self._split_list = split_list
        self._dataset_path = dataset_path
        self._split_name = split_name

    def _decode_intrinsic(self) -> SensorIntrinsic:
        if self._is_driving_subset:
            return SensorIntrinsic(fx=450.0, fy=450.0, cx=479.5, cy=269.5)
        return SensorIntrinsic(fx=1050.0, fy=1050.0, cx=479.5, cy=269.5)

    def _get_image_file_path(self) -> AnyPath:
        base_folder = get_scene_folder(
            dataset_path=self._dataset_path, scene_name=self.scene_name, split_name=self._split_name
        )
        if not self._is_full_dataset_format:
            base_folder = base_folder.parent

        return (
            base_folder / get_image_folder_name(scene_name=self.scene_name) / self.sensor_name / f"{self.frame_id}.png"
        )

    def _get_flow_file_path(self, forward: bool) -> AnyPath:
        direction = OPTICAL_FLOW_FORWARD_DIRECTION_NAME if forward else OPTICAL_FLOW_BACKWARD_DIRECTION_NAME

        suffix = "pfm"
        if self._is_full_dataset_format:
            base_folder = get_scene_flow_folder(
                dataset_path=self._dataset_path, scene_name=self.scene_name, split_name=self._split_name
            )
            annotation_path = base_folder / direction / self.sensor_name / f"{self.frame_id}.{suffix}"
        else:
            suffix = "flo"
            base_folder = get_scene_flow_folder(
                dataset_path=self._dataset_path, scene_name=self.scene_name, split_name=self._split_name
            ).parent
            annotation_path = base_folder / self.sensor_name / direction / f"{self.frame_id}.{suffix}"

        return annotation_path

    def _decode_image_dimensions(self) -> Tuple[int, int, int]:
        img_path = self._get_image_file_path()

        with img_path.open("rb") as fh:
            width, height = imagesize.get(BytesIO(fh.read()))
            return height, width, 3

    def _decode_image_rgba(self) -> np.ndarray:
        img_path = self._get_image_file_path()
        image_data = read_image(path=img_path, convert_to_rgb=True)

        ones = np.ones((*image_data.shape[:2], 1), dtype=image_data.dtype)
        concatenated = np.concatenate([image_data, ones], axis=-1)
        return concatenated

    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        return dict()

    def _decode_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        return [
            AnnotationIdentifier(annotation_type=AnnotationTypes.OpticalFlow),
            AnnotationIdentifier(annotation_type=AnnotationTypes.BackwardOpticalFlow),
        ]

    def _decode_metadata(self) -> Dict[str, Any]:
        return dict()

    def _decode_date_time(self) -> datetime:
        return frame_id_to_timestamp(frame_id=self.frame_id)

    def _decode_extrinsic(self) -> SensorExtrinsic:
        return SensorExtrinsic.from_transformation_matrix(np.eye(4))

    def _decode_sensor_pose(self) -> SensorPose:
        return SensorPose.from_transformation_matrix(np.eye(4))

    def _decode_annotations(self, identifier: AnnotationIdentifier[T]) -> T:
        if identifier.annotation_type is OpticalFlow:
            vectors = self._decode_optical_flow(forward=True)
            return OpticalFlow(vectors=vectors)
        if identifier.annotation_type is BackwardOpticalFlow:
            vectors = self._decode_optical_flow(forward=False)
            return BackwardOpticalFlow(vectors=vectors)
        else:
            raise NotImplementedError(f"{identifier.annotation_type} is not supported!")

    def _decode_optical_flow(self, forward: bool) -> np.ndarray:
        flow_file_path = self._get_flow_file_path(forward=forward)
        return read_flow(file_path=flow_file_path)

    def _decode_file_path(self, data_type: SensorDataCopyTypes) -> Optional[AnyPath]:
        if issubclass(data_type, AnnotationIdentifier):
            if data_type.annotation_type is OpticalFlow:
                return self._get_flow_file_path(forward=True)
            if data_type.annotation_type is BackwardOpticalFlow:
                return self._get_flow_file_path(forward=False)
        # Note: We also support Type[Annotation] for data_type for backwards compatibility
        elif data_type is OpticalFlow:
            return self._get_flow_file_path(forward=True)
        elif data_type is BackwardOpticalFlow:
            return self._get_flow_file_path(forward=False)

        elif issubclass(data_type, Image):
            img_path = self._get_image_file_path()
            return img_path
        return None


def read_pfm(file_path: AnyPath) -> Tuple[np.ndarray, float]:
    """
    Adapted from: https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py
    """
    with file_path.open("rb") as file:
        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file.")

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:  # little-endian
            endian = "<"
            scale = -scale
        else:
            endian = ">"  # big-endian

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data, scale


def read_flow(file_path: AnyPath) -> np.ndarray:
    """
    Adapted from: https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py
    """
    name = file_path.name
    if name.endswith(".pfm") or name.endswith(".PFM"):
        return read_pfm(file_path=file_path)[0][:, :, 0:2]
    else:
        return read_flo(path=file_path)
