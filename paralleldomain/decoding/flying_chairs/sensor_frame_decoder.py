from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import imagesize
import numpy as np

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.flying_chairs.common import frame_id_to_timestamp
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationTypes, OpticalFlow
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import SensorDataCopyTypes, SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image

T = TypeVar("T")


class FlyingChairsCameraSensorFrameDecoder(CameraSensorFrameDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        frame_id: FrameId,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        image_folder: str,
        optical_flow_folder: str,
        is_unordered_scene: bool,
        scene_decoder,
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
        self._dataset_path = dataset_path
        self._image_folder = image_folder
        self._optical_flow_folder = optical_flow_folder

    def _decode_intrinsic(self) -> SensorIntrinsic:
        return SensorIntrinsic()

    def _decode_image_dimensions(self) -> Tuple[int, int, int]:
        img_path = self._dataset_path / self._image_folder / f"{self.frame_id}"
        with img_path.open("rb") as fh:
            width, height = imagesize.get(BytesIO(fh.read()))
            return height, width, 3

    def _decode_image_rgba(self) -> np.ndarray:
        img_path = self._dataset_path / self._image_folder / f"{self.frame_id}"
        image_data = read_image(path=img_path, convert_to_rgb=True)

        ones = np.ones((*image_data.shape[:2], 1), dtype=image_data.dtype)
        concatenated = np.concatenate([image_data, ones], axis=-1)
        return concatenated

    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        return dict()

    def _decode_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        return [AnnotationIdentifier(annotation_type=AnnotationTypes.OpticalFlow)]

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
            flow_vectors = self._decode_optical_flow()
            return OpticalFlow(vectors=flow_vectors)
        else:
            raise NotImplementedError(f"{identifier.annotation_type} is not supported!")

    def _decode_optical_flow(self) -> np.ndarray:
        """
        Reads optical flow files in the .flo format. Notably used for Flying Chairs:
        https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs"""

        if self.frame_id[9] == "2":
            return None
        flow_filename = self.frame_id[:5] + "_flow.flo"
        annotation_path = self._dataset_path / self._optical_flow_folder / f"{flow_filename}"
        with annotation_path.open(mode="rb") as fp:
            header = fp.read(4)
            if header.decode("utf-8") != "PIEH":
                raise Exception("Flow file header does not contain PIEH")
            hw_raw = np.frombuffer(fp.read(8), dtype=np.int32)
            width = hw_raw[0]
            height = hw_raw[1]
            raw = np.frombuffer(fp.read(), dtype=np.float32)
            flow = raw[: (width * height) * 2].reshape((height, width, 2))
        return flow

    def _decode_file_path(self, data_type: SensorDataCopyTypes) -> Optional[AnyPath]:
        if isinstance(data_type, AnnotationIdentifier) and data_type.annotation_type is OpticalFlow:
            annotation_path = self._dataset_path / self._optical_flow_folder / f"{self.frame_id}"
            return annotation_path
        elif data_type is OpticalFlow:
            # Note: We also support Type[Annotation] for data_type for backwards compatibility
            annotation_path = self._dataset_path / self._optical_flow_folder / f"{self.frame_id}"
            return annotation_path
        elif issubclass(data_type, Image):
            img_path = self._dataset_path / self._image_folder / f"{self.frame_id}"
            return img_path
        return None
