import json
import logging
from collections import namedtuple
from datetime import datetime
from functools import lru_cache
from typing import BinaryIO, Dict, List, Optional, Type, TypeVar, Union, cast

import imageio
import numpy as np
from pyquaternion import Quaternion

from paralleldomain.decoding.decoder import Decoder
from paralleldomain.decoding.dgp_dto import (
    AnnotationsBoundingBox2DDTO,
    AnnotationsBoundingBox3DDTO,
    CalibrationDTO,
    CalibrationExtrinsicDTO,
    CalibrationIntrinsicDTO,
    DatasetDTO,
    DatasetMetaDTO,
    PoseDTO,
    SceneDataDatum,
    SceneDataDTO,
    SceneDTO,
    SceneSampleDTO,
)
from paralleldomain.model.annotation import (
    Annotation,
    AnnotationPose,
    AnnotationType,
    AnnotationTypes,
    BoundingBox2D,
    BoundingBox3D,
    BoundingBoxes2D,
    BoundingBoxes3D,
    InstanceSegmentation2D,
    InstanceSegmentation3D,
    SemanticSegmentation2D,
    SemanticSegmentation3D,
)
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.sensor import (
    ImageData,
    PointCloudData,
    SensorExtrinsic,
    SensorFrame,
    SensorIntrinsic,
    SensorPose,
)
from paralleldomain.model.transformation import Transformation
from paralleldomain.model.type_aliases import (
    AnnotationIdentifier,
    FrameId,
    SceneName,
    SensorName,
)
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.coordinate_system import (
    INTERNAL_COORDINATE_SYSTEM,
    CoordinateSystem,
)

logger = logging.getLogger(__name__)
MAX_CALIBRATIONS_TO_CACHE = 10

T = TypeVar("T")
TransformType = TypeVar("TransformType", bound=Transformation)
_DGP_TO_INTERNAL_CS = CoordinateSystem("FLU") > INTERNAL_COORDINATE_SYSTEM

_annotation_type_map: Dict[str, Type[Annotation]] = {
    "0": AnnotationTypes.BoundingBoxes2D,
    "1": AnnotationTypes.BoundingBoxes3D,
    "2": AnnotationTypes.SemanticSegmentation2D,
    "3": AnnotationTypes.SemanticSegmentation3D,
    "4": AnnotationTypes.InstanceSegmentation2D,
    "5": AnnotationTypes.InstanceSegmentation3D,
    "6": Annotation,  # Depth
    "7": Annotation,  # Surface Normals 3D
    "8": Annotation,  # Motion Vectors 2D aka Optical Flow
    "9": Annotation,  # Motion Vectors 3D aka Scene Flow
    "10": Annotation,  # Surface normals 2D
}

DGPLabel = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'Car', 'Person', ... .
        "id",  # An integer ID that is associated with this label.
        "is_thing",  # Whether this label distinguishes between single instances or not
    ],
)

_default_labels: List[DGPLabel] = [
    DGPLabel("Animal", 0, True),
    DGPLabel("Bicycle", 1, True),
    DGPLabel("Bicyclist", 2, True),
    DGPLabel("Building", 3, False),
    DGPLabel("Bus", 4, True),
    DGPLabel("Car", 5, True),
    DGPLabel("Caravan/RV", 6, True),
    DGPLabel("ConstructionVehicle", 7, True),
    DGPLabel("CrossWalk", 8, True),
    DGPLabel("Fence", 9, False),
    DGPLabel("HorizontalPole", 10, True),
    DGPLabel("LaneMarking", 11, False),
    DGPLabel("LimitLine", 12, False),
    DGPLabel("Motorcycle", 13, True),
    DGPLabel("Motorcyclist", 14, True),
    DGPLabel("OtherDriveableSurface", 15, False),
    DGPLabel("OtherFixedStructure", 16, False),
    DGPLabel("OtherMovable", 17, True),
    DGPLabel("OtherRider", 18, True),
    DGPLabel("Overpass/Bridge/Tunnel", 19, False),
    DGPLabel("OwnCar(EgoCar)", 20, False),
    DGPLabel("ParkingMeter", 21, False),
    DGPLabel("Pedestrian", 22, True),
    DGPLabel("Railway", 23, False),
    DGPLabel("Road", 24, False),
    DGPLabel("RoadBarriers", 25, False),
    DGPLabel("RoadBoundary(Curb)", 26, False),
    DGPLabel("RoadMarking", 27, False),
    DGPLabel("SideWalk", 28, False),
    DGPLabel("Sky", 29, False),
    DGPLabel("TemporaryConstructionObject", 30, True),
    DGPLabel("Terrain", 31, False),
    DGPLabel("TowedObject", 32, True),
    DGPLabel("TrafficLight", 33, True),
    DGPLabel("TrafficSign", 34, True),
    DGPLabel("Train", 35, True),
    DGPLabel("Truck", 36, True),
    DGPLabel("Vegetation", 37, False),
    DGPLabel("VerticalPole", 38, True),
    DGPLabel("WheeledSlow", 39, True),
    DGPLabel("LaneMarkingOther", 40, False),
    DGPLabel("LaneMarkingGap", 41, False),
    DGPLabel("Void", 255, False),
]

default_map = ClassMap(class_id_to_class_name={label.id: label.name for label in _default_labels})


class DGPDecoder(Decoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        max_calibrations_to_cache: int = 10,
        custom_map: Optional[ClassMap] = None,
    ):
        self.class_map = default_map if custom_map is None else custom_map

        self._dataset_path = AnyPath(dataset_path)
        self.decode_scene = lru_cache(max_calibrations_to_cache)(self.decode_scene)

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
                    f"No scene_dataset.json or file starting with scene_dataset found under {dataset_cloud_path}!"
                )
            scene_json_path: AnyPath = dataset_cloud_path / files_with_prefix[-1]

        with scene_json_path.open(mode="r") as f:
            scene_dataset = json.load(f)

        meta_data = DatasetMetaDTO.from_dict(scene_dataset["metadata"])
        scene_names: List[str] = scene_dataset["scene_splits"]["0"]["filenames"]
        return DatasetDTO(meta_data=meta_data, scene_names=scene_names)

    def decode_scene(self, scene_name: str) -> SceneDTO:
        scene_folder = self._dataset_path / scene_name
        potential_scene_files = [
            name.name for name in scene_folder.iterdir() if name.name.startswith("scene") and name.name.endswith("json")
        ]

        if len(potential_scene_files) == 0:
            logger.error(f"No sceneXXX.json found under {scene_folder}!")

        scene_file = scene_folder / potential_scene_files[0]
        with scene_file.open("r") as f:
            scene_data = json.load(f)
            scene_dto = SceneDTO.from_dict(scene_data)
            return scene_dto

    def decode_calibration(self, scene_name: str, calibration_key: str) -> CalibrationDTO:
        calibration_path = self._dataset_path / scene_name / "calibration" / f"{calibration_key}.json"
        with calibration_path.open("r") as f:
            cal_dict = json.load(f)
            return CalibrationDTO.from_dict(cal_dict)

    def decode_extrinsic_calibration(
        self, scene_name: str, calibration_key: str, sensor_name: SensorName
    ) -> CalibrationExtrinsicDTO:
        calibration_dto = self.decode_calibration(scene_name=scene_name, calibration_key=calibration_key)
        index = calibration_dto.names.index(sensor_name)
        return calibration_dto.extrinsics[index]

    def decode_intrinsic_calibration(
        self, scene_name: str, calibration_key: str, sensor_name: SensorName
    ) -> CalibrationIntrinsicDTO:
        calibration_dto = self.decode_calibration(scene_name=scene_name, calibration_key=calibration_key)
        index = calibration_dto.names.index(sensor_name)
        return calibration_dto.intrinsics[index]

    def decode_bounding_boxes_3d(self, scene_name: str, annotation_identifier: str) -> AnnotationsBoundingBox3DDTO:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        with annotation_path.open("r") as f:
            return AnnotationsBoundingBox3DDTO.from_dict(json.load(f))

    def decode_bounding_boxes_2d(self, scene_name: str, annotation_identifier: str) -> AnnotationsBoundingBox2DDTO:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        print(annotation_path)
        with annotation_path.open("r") as f:
            return AnnotationsBoundingBox2DDTO.from_dict(json.load(f))

    def decode_semantic_segmentation_3d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        with annotation_path.open(mode="rb") as cloud_binary:
            npz_data = np.load(cast(BinaryIO, cloud_binary))
            return npz_data.f.segmentation

    def decode_instance_segmentation_3d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        with annotation_path.open(mode="rb") as cloud_binary:
            npz_data = np.load(cast(BinaryIO, cloud_binary))
            return npz_data.f.instance

    def decode_semantic_segmentation_2d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        with annotation_path.open(mode="rb") as cloud_binary:
            image_data = np.asarray(imageio.imread(cast(BinaryIO, cloud_binary), format="png"))
            return image_data

    def decode_instance_segmentation_2d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        with annotation_path.open(mode="rb") as cloud_binary:
            image_data = np.asarray(imageio.imread(cast(BinaryIO, cloud_binary), format="png"))
            return image_data

    def decode_point_cloud(self, scene_name: str, cloud_identifier: str) -> np.ndarray:
        cloud_path = self._dataset_path / scene_name / cloud_identifier
        with cloud_path.open(mode="rb") as cloud_binary:
            npz_data = np.load(cast(BinaryIO, cloud_binary))
            pc_data = npz_data.f.data
            return np.column_stack([pc_data[c] for c in pc_data.dtype.names])

    def decode_image_rgb(self, scene_name: str, cloud_identifier: str) -> np.ndarray:
        cloud_path = self._dataset_path / scene_name / cloud_identifier
        with cloud_path.open(mode="rb") as cloud_binary:
            image_data = np.asarray(imageio.imread(cast(BinaryIO, cloud_binary), format="png"))
            return image_data

    # ------------------------------------------------
    def get_unique_scene_id(self, scene_name: SceneName) -> str:
        return f"{self._dataset_path}-{scene_name}"

    def decode_scene_names(self) -> List[SceneName]:
        dto = self.decode_dataset()
        return [AnyPath(path).parent.name for path in dto.scene_names]

    def decode_dataset_meta_data(self) -> DatasetMeta:
        dto = self.decode_dataset()
        meta_dict = dto.meta_data.to_dict()
        anno_types = [_annotation_type_map[str(a)] for a in dto.meta_data.available_annotation_types]
        return DatasetMeta(name=dto.meta_data.name, available_annotation_types=anno_types, custom_attributes=meta_dict)

    def decode_scene_description(self, scene_name: SceneName) -> str:
        scene_dto = self.decode_scene(scene_name=scene_name)
        return scene_dto.description

    def decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, datetime]:
        scene_dto = self.decode_scene(scene_name=scene_name)
        return {sample.id.index: self._scene_sample_to_date_time(sample=sample) for sample in scene_dto.samples}

    def decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        scene_dto = self.decode_scene(scene_name=scene_name)
        return list({datum.id.name for datum in scene_dto.data})

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
        unique_cache_key = f"{self._dataset_path}-{scene_name}-{frame_id}-{sensor_name}"
        sensor_frame = SensorFrame(
            unique_cache_key=unique_cache_key,
            frame_id=frame_id,
            date_time=self._scene_sample_to_date_time(sample=sample),
            sensor_name=sensor_name,
            lazy_loader=_FrameLazyLoader(
                unique_cache_key_prefix=unique_cache_key,
                decoder=self,
                class_map=self.class_map,
                scene_name=scene_name,
                sensor_name=sensor_name,
                calibration_key=sample.calibration_key,
                datum=scene_data.datum,
            ),
        )
        return sensor_frame

    @staticmethod
    def _scene_sample_to_date_time(sample: SceneSampleDTO) -> datetime:
        return datetime.strptime(sample.id.timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")


class _FrameLazyLoader:
    def __init__(
        self,
        unique_cache_key_prefix: str,
        decoder: DGPDecoder,
        scene_name: SceneName,
        sensor_name: SensorName,
        class_map: ClassMap,
        calibration_key: str,
        datum: SceneDataDatum,
    ):
        self.class_map = class_map
        self.datum = datum
        self._unique_cache_key_prefix = unique_cache_key_prefix
        self.sensor_name = sensor_name
        self.scene_name = scene_name
        self.decoder = decoder
        self.calibration_key = calibration_key

    def load_intrinsic(self) -> SensorIntrinsic:
        dto = self.decoder.decode_intrinsic_calibration(
            scene_name=self.scene_name,
            calibration_key=self.calibration_key,
            sensor_name=self.sensor_name,
        )

        if dto.fisheye is True:
            camera_model = "fisheye"
        elif dto.fisheye is False:
            camera_model = "brown_conrady"
        elif dto.fisheye > 1:
            camera_model = f"custom_{dto.fisheye}"

        return SensorIntrinsic(
            cx=dto.cx,
            cy=dto.cy,
            fx=dto.fx,
            fy=dto.fy,
            k1=dto.k1,
            k2=dto.k2,
            p1=dto.p1,
            p2=dto.p2,
            k3=dto.k3,
            k4=dto.k4,
            k5=dto.k5,
            k6=dto.k6,
            skew=dto.skew,
            fov=dto.fov,
            camera_model=camera_model,
        )

    def load_extrinsic(self) -> SensorExtrinsic:
        dto = self.decoder.decode_extrinsic_calibration(
            scene_name=self.scene_name,
            calibration_key=self.calibration_key,
            sensor_name=self.sensor_name,
        )
        return _pose_dto_to_transformation(dto=dto, transformation_type=SensorExtrinsic)

    def load_point_cloud(self) -> Optional[PointCloudData]:
        if self.datum.point_cloud:
            unique_cache_key = f"{self._unique_cache_key_prefix}-point_cloud"
            return PointCloudData(
                unique_cache_key=unique_cache_key,
                point_format=self.datum.point_cloud.point_format,
                load_data=lambda: self.decoder.decode_point_cloud(
                    scene_name=self.scene_name, cloud_identifier=self.datum.point_cloud.filename
                ),
            )
        return None

    def load_image(self) -> Optional[ImageData]:
        if self.datum.image:
            return ImageData(
                load_data_rgba=lambda: self.decoder.decode_image_rgb(
                    scene_name=self.scene_name,
                    cloud_identifier=self.datum.image.filename,
                ),
            )

    def load_sensor_pose(self) -> SensorPose:
        if self.datum.image:
            return _pose_dto_to_transformation(dto=self.datum.image.pose, transformation_type=SensorPose)
        else:
            return _pose_dto_to_transformation(dto=self.datum.point_cloud.pose, transformation_type=SensorPose)

    def load_annotations(self, identifier: AnnotationIdentifier, annotation_type: Type[T]) -> T:
        if issubclass(annotation_type, BoundingBoxes3D):
            dto = self.decoder.decode_bounding_boxes_3d(scene_name=self.scene_name, annotation_identifier=identifier)

            box_list = []
            for box_dto in dto.annotations:
                pose = _pose_dto_to_transformation(dto=box_dto.box.pose, transformation_type=AnnotationPose)
                box = BoundingBox3D(
                    pose=pose,
                    width=box_dto.box.width,
                    length=box_dto.box.length,
                    height=box_dto.box.width,
                    class_id=box_dto.class_id,
                    instance_id=box_dto.instance_id,
                    num_points=box_dto.num_points,
                )
                box_list.append(box)

            return BoundingBoxes3D(boxes=box_list, class_map=self.class_map)
        elif issubclass(annotation_type, BoundingBoxes2D):
            dto = self.decoder.decode_bounding_boxes_2d(scene_name=self.scene_name, annotation_identifier=identifier)

            box_list = []
            for box_dto in dto.annotations:
                user_data = json.loads(box_dto.attributes.user_data)

                box = BoundingBox2D(
                    x=box_dto.box.x,
                    y=box_dto.box.y,
                    width=box_dto.box.w,
                    height=box_dto.box.h,
                    class_id=box_dto.class_id,
                    instance_id=box_dto.instance_id,
                    visibility=float(user_data["visibility"]),
                )
                box_list.append(box)

            return BoundingBoxes2D(boxes=box_list, class_map=self.class_map)
        elif issubclass(annotation_type, SemanticSegmentation3D):
            segmentation_mask = self.decoder.decode_semantic_segmentation_3d(
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            return SemanticSegmentation3D(mask=segmentation_mask, class_map=self.class_map)
        elif issubclass(annotation_type, InstanceSegmentation3D):
            instance_mask = self.decoder.decode_instance_segmentation_3d(
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            return InstanceSegmentation3D(mask=instance_mask)
        elif issubclass(annotation_type, SemanticSegmentation2D):
            segmentation_mask = self.decoder.decode_semantic_segmentation_2d(
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            return SemanticSegmentation2D(mask=segmentation_mask, class_map=self.class_map)
        elif issubclass(annotation_type, InstanceSegmentation2D):
            segmentation_mask = self.decoder.decode_instance_segmentation_2d(
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            return InstanceSegmentation2D(mask=segmentation_mask)

    def load_available_annotation_types(
        self,
    ) -> Dict[AnnotationType, AnnotationIdentifier]:
        if self.datum.image:
            type_to_path = self.datum.image.annotations
        else:
            type_to_path = self.datum.point_cloud.annotations
        return {_annotation_type_map[k]: v for k, v in type_to_path.items()}


def _pose_dto_to_transformation(dto: PoseDTO, transformation_type: Type[TransformType]) -> TransformType:
    transform = transformation_type(
        quaternion=Quaternion(dto.rotation.qw, dto.rotation.qx, dto.rotation.qy, dto.rotation.qz),
        translation=np.array([dto.translation.x, dto.translation.y, dto.translation.z]),
    )
    return transformation_type.from_transformation_matrix(_DGP_TO_INTERNAL_CS @ transform.transformation_matrix)
