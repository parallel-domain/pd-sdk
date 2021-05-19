import json
from functools import lru_cache
from typing import Union, List, cast, BinaryIO, Dict, Optional, Type, TypeVar
import logging

from pyquaternion import Quaternion

import numpy as np
from paralleldomain.model.annotation import Annotation, AnnotationType, AnnotationTypes, BoundingBox3D, AnnotationPose
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.decoding.decoder import Decoder
from paralleldomain.decoding.dgp_dto import DatasetDTO, DatasetMetaDTO, SceneDTO, CalibrationDTO, \
    AnnotationsBoundingBox3DDTO, \
    CalibrationExtrinsicDTO, CalibrationIntrinsicDTO, SceneDataDTO, SceneSampleDTO, PoseDTO, SceneDataDatum
from paralleldomain.model.transformation import Transformation
from paralleldomain.model.sensor import PointCloudData, SensorFrame, SensorPose, SensorExtrinsic, SensorIntrinsic
from paralleldomain.model.type_aliases import SensorName, SceneName, FrameId, AnnotationIdentifier
from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger(__name__)
MAX_CALIBRATIONS_TO_CACHE = 10

T = TypeVar('T')
TransformType = TypeVar('TransformType')

_annotation_type_map: Dict[str, Type[Annotation]] = {
    "0": AnnotationTypes.BoundingBox2D,
    "1": AnnotationTypes.BoundingBox3D,
    "2": AnnotationTypes.SemanticSegmentation2D,
    "3": Annotation,
    "4": Annotation,
    "5": Annotation,
    "6": Annotation,
    "7": Annotation,
    "8": Annotation,
    "9": Annotation,
    "10": Annotation,
}


class DGPDecoder(Decoder):
    def __init__(self, dataset_path: Union[str, AnyPath], max_calibrations_to_cache: int = 10,
                 max_point_clouds_to_cache: int = 50, max_annotations_to_cache: int = 50):
        self._dataset_path = AnyPath(dataset_path)
        self.decode_scene = lru_cache(max_calibrations_to_cache)(self.decode_scene)
        self.decode_point_cloud = lru_cache(max_point_clouds_to_cache)(self.decode_point_cloud)
        self.decode_3d_bounding_boxes = lru_cache(max_annotations_to_cache)(self.decode_3d_bounding_boxes)

    @lru_cache(maxsize=1)
    def _data_by_key(self, scene_name: str) -> Dict[str, SceneDataDTO]:
        dto = self.decode_scene(scene_name=scene_name)
        return {d.key: d for d in dto.data}

    @lru_cache(maxsize=1)
    def _data_by_key_with_name(self, scene_name: str, data_name: str) -> Dict[str, SceneDataDTO]:
        dto = self.decode_scene(scene_name=scene_name)
        return {d.key: d for d in dto.data if d.id.name == data_name}

    @lru_cache(maxsize=1)
    def _sample_by_index(self, scene_name: str) -> Dict[str, SceneSampleDTO]:
        dto = self.decode_scene(scene_name=scene_name)
        return {s.id.index: s for s in dto.samples}

    @lru_cache(maxsize=1)
    def decode_dataset(self) -> DatasetDTO:
        dataset_cloud_path: AnyPath = AnyPath(self._dataset_path)
        scene_json_path: AnyPath = dataset_cloud_path / "scene_dataset.json"
        if not scene_json_path.exists():
            files_with_prefix = [name.name for name in dataset_cloud_path.iterdir() if "scene_dataset" in name.name]
            if len(files_with_prefix) == 0:
                logger.error(
                    f"No scene_dataset.json or file starting with scene_dataset found under {dataset_cloud_path}!")
            scene_json_path: AnyPath = dataset_cloud_path / files_with_prefix[-1]

        with scene_json_path.open(mode="r") as f:
            scene_dataset = json.load(f)

        meta_data = DatasetMetaDTO.from_dict(scene_dataset["metadata"])
        scene_names: List[str] = scene_dataset["scene_splits"]["0"]["filenames"]
        return DatasetDTO(meta_data=meta_data, scene_names=scene_names)

    @lru_cache(maxsize=1)
    def decode_scene(self, scene_name: str) -> SceneDTO:
        with (self._dataset_path / scene_name).open("r") as f:
            scene_data = json.load(f)
            scene_dto = SceneDTO.from_dict(scene_data)
            return scene_dto

    def decode_calibration(self, scene_name: str, calibration_key: str) -> CalibrationDTO:
        calibration_path = self._dataset_path / scene_name / "calibration" / f"{calibration_key}.json"
        with calibration_path.open("r") as f:
            cal_dict = json.load(f)
            return CalibrationDTO.from_dict(cal_dict)

    def decode_extrinsic_calibration(self, scene_name: str, calibration_key: str, sensor_name: SensorName) \
            -> CalibrationExtrinsicDTO:
        calibration_dto = self.decode_calibration(scene_name=scene_name, calibration_key=calibration_key)
        index = calibration_dto.names.index(sensor_name)
        return calibration_dto.extrinsics[index]

    def decode_intrinsic_calibration(self, scene_name: str, calibration_key: str, sensor_name: SensorName) \
            -> CalibrationIntrinsicDTO:
        calibration_dto = self.decode_calibration(scene_name=scene_name, calibration_key=calibration_key)
        index = calibration_dto.names.index(sensor_name)
        return calibration_dto.intrinsics[index]

    def decode_3d_bounding_boxes(self, scene_name: str, annotation_identifier: str) -> AnnotationsBoundingBox3DDTO:
        annotation_path = (self._dataset_path / scene_name).parent / annotation_identifier
        with annotation_path.open("r") as f:
            return AnnotationsBoundingBox3DDTO.from_dict(json.load(f))

    def decode_point_cloud(self, scene_name: str, cloud_identifier: str, num_channels: int) -> np.ndarray:
        cloud_path = (self._dataset_path / scene_name).parent / cloud_identifier
        with cloud_path.open(mode="rb") as cloud_binary:
            npz_data = np.load(cast(BinaryIO, cloud_binary))
        return np.array([f.tolist() for f in npz_data.f.data]).reshape(-1, num_channels)

    # ------------------------------------------------
    def decode_scene_names(self) -> List[SceneName]:
        dto = self.decode_dataset()
        return dto.scene_names

    def decode_dataset_meta_data(self) -> DatasetMeta:
        dto = self.decode_dataset()
        return DatasetMeta(**dto.meta_data.to_dict())

    def decode_scene_description(self, scene_name: SceneName) -> str:
        scene_dto = self.decode_scene(scene_name=scene_name)
        return scene_dto.description

    def decode_frame_ids(self, scene_name: SceneName) -> List[FrameId]:
        scene_dto = self.decode_scene(scene_name=scene_name)
        return [sample.id.index for sample in scene_dto.samples]

    def decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        scene_dto = self.decode_scene(scene_name=scene_name)
        return list(set([datum.id.name for datum in scene_dto.data]))

    def decode_available_sensor_names(self, scene_name: SceneName, frame_id: FrameId) -> List[SensorName]:
        # sample of current frame
        sample = self._sample_by_index(scene_name=scene_name)[frame_id]
        # all sensor data of the sensor
        sensor_data = self._data_by_key(scene_name=scene_name)
        return [sensor_data[key].id.name for key in sample.datum_keys]

    def decode_sensor_frame(self, scene_name: SceneName, frame_id: FrameId, sensor_name: SensorName) -> SensorFrame:
        # sample of current frame
        sample = self._sample_by_index(scene_name=scene_name)[frame_id]
        # all sensor data of the sensor
        sensor_data = self._data_by_key_with_name(scene_name=scene_name, data_name=sensor_name)
        # datum ley of sample that references the given sensor name
        datum_key = next(iter([key for key in sample.datum_keys if key in sensor_data]))
        scene_data = sensor_data[datum_key]

        sensor_frame = SensorFrame(
            sensor_name=sensor_name,
            lazy_loader=_FrameLazyLoader(decoder=self, scene_name=scene_name,
                                         sensor_name=sensor_name, calibration_key=sample.calibration_key,
                                         datum=scene_data.datum))
        return sensor_frame


class _FrameLazyLoader:
    def __init__(self, decoder: DGPDecoder, scene_name: SceneName, sensor_name: SensorName,
                 calibration_key: str, datum: SceneDataDatum):
        self.datum = datum
        self.sensor_name = sensor_name
        self.scene_name = scene_name
        self.decoder = decoder
        self.calibration_key = calibration_key

    def load_intrinsic(self) -> SensorIntrinsic:
        dto = self.decoder.decode_intrinsic_calibration(scene_name=self.scene_name,
                                                        calibration_key=self.calibration_key,
                                                        sensor_name=self.sensor_name)
        return SensorIntrinsic(cx=dto.cx, cy=dto.cy, fx=dto.fx, fy=dto.fy, k1=dto.k1, k2=dto.k2, p1=dto.p1,
                               p2=dto.p2, k3=dto.k3, k4=dto.k4, k5=dto.k5, k6=dto.k6, skew=dto.skew, fov=dto.fov,
                               fisheye=dto.fisheye)

    def load_extrinsic(self) -> SensorExtrinsic:
        dto = self.decoder.decode_extrinsic_calibration(
            scene_name=self.scene_name,
            calibration_key=self.calibration_key,
            sensor_name=self.sensor_name)
        return _post_dto_to_transformation(dto=dto, transformation_type=SensorExtrinsic)

    def load_point_cloud(self) -> Optional[PointCloudData]:
        if self.datum.point_cloud:
            return PointCloudData(point_format=self.datum.point_cloud.point_format,
                                  load_data=lambda: self.decoder.decode_point_cloud(
                                      scene_name=self.scene_name,
                                      cloud_identifier=self.datum.point_cloud.filename,
                                      num_channels=len(self.datum.point_cloud.point_format)))
        return None

    def load_sensor_pose(self) -> SensorPose:
        if self.datum.image:
            return _post_dto_to_transformation(dto=self.datum.image.pose, transformation_type=SensorPose)
        else:
            return _post_dto_to_transformation(dto=self.datum.image.pose, transformation_type=SensorPose)

    def load_annotations(self, identifier: AnnotationIdentifier, annotation_type: Type[T]) -> List[T]:

        annotations = list()
        if issubclass(annotation_type, BoundingBox3D):
            dto = self.decoder.decode_3d_bounding_boxes(scene_name=self.scene_name,
                                                        annotation_identifier=identifier)
            for box_dto in dto.annotations:
                pose = _post_dto_to_transformation(dto=box_dto.box.pose, transformation_type=AnnotationPose)
                box = BoundingBox3D(
                    pose=pose,
                    width=box_dto.box.width,
                    length=box_dto.box.length,
                    height=box_dto.box.width,
                    class_id=box_dto.class_id,
                    instance_id=box_dto.instance_id,
                    num_points=box_dto.num_points)
                annotations.append(box)

        return annotations

    def load_available_annotation_types(self) -> Dict[AnnotationType, AnnotationIdentifier]:
        if self.datum.image:
            type_to_path = self.datum.image.annotations
        else:
            type_to_path = self.datum.point_cloud.annotations
        return {_annotation_type_map[k]: v for k, v in type_to_path.items()}


def _post_dto_to_transformation(dto: PoseDTO, transformation_type: Type[TransformType]) -> TransformType:
    tf = transformation_type(quaternion=Quaternion(dto.rotation.qw, dto.rotation.qx, dto.rotation.qy, dto.rotation.qz,),
                             translation=np.array([dto.translation.x, dto.translation.y, dto.translation.z]))
    # tf.rotation_quaternion = [
    #     dto.rotation.qw,
    #     dto.rotation.qx,
    #     dto.rotation.qy,
    #     dto.rotation.qz,
    # ]
    # tf.translation = [
    #     dto.translation.x,
    #     dto.translation.y,
    #     dto.translation.z,
    # ]
    return tf
