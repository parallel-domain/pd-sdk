import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union, overload

from pd.data_lab import LabeledStateReference
from pd.label_engine import DEFAULT_LABEL_ENGINE_CONFIG_NAME, load_pipeline_config

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.data_stream import INSTANCE_POINT_COLOR_MAP
from paralleldomain.decoding.data_stream.data_accessor import (
    DataStreamDataAccessor,
    LabelEngineDataStreamDataAccessor,
    StoredBatchDataStreamDataAccessor,
    StoredDataStreamDataAccessor,
)
from paralleldomain.decoding.data_stream.frame_decoder import DataStreamFrameDecoder
from paralleldomain.decoding.data_stream.sensor_decoder import DataStreamCameraSensorDecoder
from paralleldomain.decoding.decoder import DatasetDecoder, SceneDecoder, TDateTime
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder, RadarSensorDecoder
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationType, AnnotationTypes
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

StreamType = int

_NODE_TYPE_TO_ANNOTATION_TYPE_MAP = {
    "type.googleapis.com/pd.data.GenerateBoundingBox3DConfig": AnnotationTypes.BoundingBoxes3D,
    "type.googleapis.com/pd.data.BoundingBox2DConfig": AnnotationTypes.BoundingBoxes2D,
    "type.googleapis.com/pd.data.GenerateInstanceMaskConfig": AnnotationTypes.InstanceSegmentation2D,
    "type.googleapis.com/pd.data.GenerateSemanticMaskConfig": AnnotationTypes.SemanticSegmentation2D,
    "type.googleapis.com/pd.data.GenerateProjectionConfig": AnnotationTypes.Points2D,
    "type.googleapis.com/pd.data.InstancePoint3DAnnotatorConfig": AnnotationTypes.Points3D,
}


class DataStreamDatasetDecoder(DatasetDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        settings: Optional[DecoderSettings] = None,
        camera_image_stream_name: str = "rgb",
        available_annotation_identifiers: Optional[List[AnnotationIdentifier]] = None,
        label_engine_config_name: Optional[str] = DEFAULT_LABEL_ENGINE_CONFIG_NAME,
    ):
        self._init_kwargs = dict(
            dataset_path=dataset_path,
            settings=settings,
        )
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        self._camera_image_stream_name = camera_image_stream_name
        self._available_annotation_identifiers = available_annotation_identifiers
        self._label_engine_config_name = label_engine_config_name

        dataset_name = "-".join([str(dataset_path)])
        super().__init__(dataset_name=dataset_name, settings=settings)

    def create_scene_decoder(self, scene_name: SceneName) -> "SceneDecoder":
        scene_path = self._dataset_path / scene_name
        if not scene_path.exists() and scene_path.is_dir():
            raise ValueError(f"Can't create decoder for scene {scene_name}: {scene_path} does not exist!")
        return DataStreamSceneDecoder(
            dataset_name=self.dataset_name,
            settings=self.settings,
            scene_path=scene_path,
            scene_name=scene_name,
            camera_image_stream_name=self._camera_image_stream_name,
            available_annotation_identifiers=self._available_annotation_identifiers,
            label_engine_config_name=self._label_engine_config_name,
        )

    def _decode_unordered_scene_names(self) -> List[SceneName]:
        return self._decode_scene_names()

    def _decode_scene_names(self) -> List[SceneName]:
        return [f.stem for f in self._dataset_path.iterdir() if f.is_dir()]

    def _decode_dataset_metadata(self) -> DatasetMeta:
        return DatasetMeta(
            name=self.dataset_name,
            available_annotation_identifiers=self._available_annotation_identifiers or [],
            custom_attributes=dict(),
        )

    @staticmethod
    def get_format() -> str:
        return "data-stream"

    def get_path(self) -> Optional[AnyPath]:
        return self._dataset_path

    def get_decoder_init_kwargs(self) -> Dict[str, Any]:
        return self._init_kwargs


class DataStreamSceneDecoder(SceneDecoder[datetime]):
    @overload
    def __init__(
        self,
        *,
        dataset_name: str,
        settings: DecoderSettings,
        scene_name: SceneName,
        scene_path: AnyPath,
        available_annotation_identifiers: Optional[List[AnnotationIdentifier]] = None,
        camera_image_stream_name: str = "rgb",
        label_engine_config_name: Optional[str] = DEFAULT_LABEL_ENGINE_CONFIG_NAME,
    ):
        ...

    @overload
    def __init__(
        self,
        *,
        dataset_name: str,
        scene_name: SceneName,
        settings: DecoderSettings,
        state_reference: LabeledStateReference,
        available_annotation_identifiers: List[AnnotationIdentifier],
        camera_image_stream_name: str = "rgb",
        label_engine_config_name: Optional[str] = None,
    ):
        ...

    def __init__(
        self,
        *,
        dataset_name: str,
        scene_name: SceneName,
        settings: DecoderSettings,
        label_engine_config_name: Optional[str] = None,
        state_reference: Optional[LabeledStateReference] = None,
        scene_path: Optional[AnyPath] = None,
        available_annotation_identifiers: Optional[List[AnnotationIdentifier]] = None,
        camera_image_stream_name: str = "rgb",
    ):
        super().__init__(dataset_name=dataset_name, settings=settings, scene_name=scene_name)
        self._scene_path = scene_path
        self._data_accessor = self._resolve_data_accessor(
            scene_name=scene_name,
            label_engine_config_name=label_engine_config_name,
            state_reference=state_reference,
            scene_path=scene_path,
            available_annotation_identifiers=available_annotation_identifiers,
            camera_image_stream_name=camera_image_stream_name,
        )

    @staticmethod
    def _resolve_data_accessor(
        scene_name: SceneName,
        label_engine_config_name: Optional[str] = None,
        state_reference: Optional[LabeledStateReference] = None,
        scene_path: Optional[AnyPath] = None,
        available_annotation_identifiers: Optional[List[AnnotationIdentifier]] = None,
        camera_image_stream_name: str = "rgb",
    ) -> DataStreamDataAccessor:
        if state_reference is not None:
            label_engine_config_name = state_reference.label_engine.config_name
        label_engine_pipeline_config = json.loads(load_pipeline_config(label_engine_config_name))
        ontology_stream_name = DataStreamSceneDecoder._resolve_ontology_stream_name(
            label_engine_pipeline_config=label_engine_pipeline_config
        )
        output_path_to_generator_type = {
            node["config"]["@type"]: node["config"]["output_path"] for node in label_engine_pipeline_config["nodes"]
        }
        config_annotation_identifiers = [
            AnnotationIdentifier(annotation_type=_NODE_TYPE_TO_ANNOTATION_TYPE_MAP[k], name=v)
            for k, v in output_path_to_generator_type.items()
            if k in _NODE_TYPE_TO_ANNOTATION_TYPE_MAP
        ] + [AnnotationIdentifier(annotation_type=AnnotationTypes.Depth, name="depth")]

        if scene_path is not None:
            if available_annotation_identifiers is None:
                available_annotation_identifiers = config_annotation_identifiers
                fixed_annotation_identifiers = [
                    AnnotationIdentifier(annotation_type=AnnotationTypes.Albedo2D, name="base"),
                    AnnotationIdentifier(annotation_type=AnnotationTypes.MaterialProperties2D, name="material"),
                    AnnotationIdentifier(
                        annotation_type=AnnotationTypes.MaterialProperties3D, name="material_properties_3d"
                    ),
                    AnnotationIdentifier(annotation_type=AnnotationTypes.SurfaceNormals2D, name="normals"),
                    AnnotationIdentifier(annotation_type=AnnotationTypes.SurfaceNormals3D, name="surface_normals_3d"),
                    AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D, name="bounding_box_2d"),
                    AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D, name="bounding_box_2d_xyz"),
                    AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes3D, name="bounding_box_3d"),
                    AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes3D, name="bounding_box_3d_xyz"),
                    AnnotationIdentifier(annotation_type=AnnotationTypes.Points2D, name="instance_points_2d"),
                    AnnotationIdentifier(annotation_type=AnnotationTypes.Points2D, name="instance_points_2d_xyz"),
                    AnnotationIdentifier(annotation_type=AnnotationTypes.Points3D, name="instance_points_3d"),
                    AnnotationIdentifier(annotation_type=AnnotationTypes.Points3D, name="instance_points_3d_xyz"),
                    AnnotationIdentifier(annotation_type=AnnotationTypes.SemanticSegmentation2D, name="semantic_mask"),
                    AnnotationIdentifier(
                        annotation_type=AnnotationTypes.SemanticSegmentation2D, name="semantic_mask_xyz"
                    ),
                    AnnotationIdentifier(annotation_type=AnnotationTypes.InstanceSegmentation2D, name="instance_mask"),
                    AnnotationIdentifier(
                        annotation_type=AnnotationTypes.InstanceSegmentation2D, name="instance_mask_xyz"
                    ),
                ]
                fixed_annotation_identifiers = [
                    f for f in fixed_annotation_identifiers if f not in available_annotation_identifiers
                ]
                available_annotation_identifiers += fixed_annotation_identifiers

            mesh_map_stream = scene_path / "mesh_map"
            if mesh_map_stream.exists() and len(list(mesh_map_stream.iterdir())) == 2:
                # batch mode has a mesh map with a single file + type file
                # data lab mode has one per timestamp
                # this fails for single frame datasets
                data_accessor: DataStreamDataAccessor = StoredBatchDataStreamDataAccessor(
                    scene_path=scene_path,
                    scene_name=scene_name,
                    camera_image_stream_name=camera_image_stream_name,
                    potentially_available_annotation_identifiers=available_annotation_identifiers,
                    ontology_stream_name=ontology_stream_name,
                )
            else:
                data_accessor: DataStreamDataAccessor = StoredDataStreamDataAccessor(
                    scene_path=scene_path,
                    scene_name=scene_name,
                    camera_image_stream_name=camera_image_stream_name,
                    potentially_available_annotation_identifiers=available_annotation_identifiers,
                    ontology_stream_name=ontology_stream_name,
                )
        else:
            if available_annotation_identifiers is None:
                available_annotation_identifiers = config_annotation_identifiers
                if any([sensor.capture_normals for sensor in state_reference.sensor_rig]):
                    available_annotation_identifiers.append(
                        AnnotationIdentifier(annotation_type=AnnotationTypes.SurfaceNormals2D, name="normals")
                    )
                if any([sensor.capture_basecolor for sensor in state_reference.sensor_rig]):
                    available_annotation_identifiers.append(
                        AnnotationIdentifier(annotation_type=AnnotationTypes.Albedo2D, name="base")
                    )
                if any([sensor.capture_properties for sensor in state_reference.sensor_rig]):
                    available_annotation_identifiers.append(
                        AnnotationIdentifier(annotation_type=AnnotationTypes.MaterialProperties2D, name="material")
                    )
            data_accessor = LabelEngineDataStreamDataAccessor(
                labeled_state_reference=state_reference,
                scene_name=scene_name,
                camera_image_stream_name=camera_image_stream_name,
                available_annotation_identifiers=available_annotation_identifiers,
                ontology_stream_name=ontology_stream_name,
            )
        return data_accessor

    def update_labeled_state_reference(self, labeled_state_reference: LabeledStateReference) -> None:
        if not isinstance(self._data_accessor, LabelEngineDataStreamDataAccessor):
            raise ValueError("Can only update labeled state reference on LabelEngineDataStreamDataAccessor")
        self._data_accessor.update_labeled_state_reference(labeled_state_reference)

    def _decode_set_metadata(self) -> Dict[str, Any]:
        return self._data_accessor.get_scene_metadata()

    def _decode_set_description(self) -> str:
        return ""

    def _decode_frame_id_set(self) -> Set[FrameId]:
        return self._data_accessor.get_frame_ids()

    def _decode_sensor_names(self) -> List[SensorName]:
        return list({s for fid, sensors in self._data_accessor.sensors.items() for s in sensors.keys()})

    def _decode_camera_names(self) -> List[SensorName]:
        return list({s for fid, sensors in self._data_accessor.cameras.items() for s in sensors.keys()})

    def _decode_lidar_names(self) -> List[SensorName]:
        return list({s for fid, sensors in self._data_accessor.lidars.items() for s in sensors.keys()})

    def _decode_radar_names(self) -> List[SensorName]:
        return list({s for fid, sensors in self._data_accessor.radars.items() for s in sensors.keys()})

    def _decode_class_maps(self) -> Dict[AnnotationType, ClassMap]:
        label_data = self._data_accessor.get_ontology_data(frame_id=next(iter(self.get_frame_ids())))
        ontology = label_data.data_as_semantic_label_map
        semantic_label_map = ontology.semantic_label_map
        pd_class_details = []
        for semantic_id in semantic_label_map:
            c = semantic_label_map[semantic_id]
            class_detail = ClassDetail(
                name=c.label,
                id=int(c.id),
                instanced=False,  # TODO deprecate this parameter
                meta=dict(supercategory="", color={"r": c.color.red, "g": c.color.green, "b": c.color.blue}),
            )
            pd_class_details.append(class_detail)

        class_maps = {
            identifier: ClassMap(classes=pd_class_details) for identifier in self.get_available_annotation_identifiers()
        }
        return class_maps

    def _create_camera_sensor_decoder(self, sensor_name: SensorName) -> CameraSensorDecoder[TDateTime]:
        return DataStreamCameraSensorDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=sensor_name,
            settings=self.settings,
            data_accessor=self._data_accessor,
            is_unordered_scene=False,
            scene_decoder=self,
        )

    def _create_lidar_sensor_decoder(self, sensor_name: SensorName) -> LidarSensorDecoder[TDateTime]:
        raise NotImplementedError("Lidar decoding not implemented")

    def _create_radar_sensor_decoder(self, radar_name: SensorName) -> RadarSensorDecoder[TDateTime]:
        raise NotImplementedError("Radar decoding not implemented")

    def _create_frame_decoder(self, frame_id: FrameId) -> FrameDecoder[TDateTime]:
        return DataStreamFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            frame_id=frame_id,
            settings=self.settings,
            data_accessor=self._data_accessor,
            is_unordered_scene=False,
            scene_decoder=self,
        )

    def _decode_frame_id_to_date_time_map(self) -> Dict[FrameId, TDateTime]:
        return self._data_accessor.get_frame_id_to_date_time_map()

    def _decode_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        return self._data_accessor.available_annotation_identifiers

    @staticmethod
    def _resolve_ontology_stream_name(label_engine_pipeline_config: dict) -> str:
        potential_ontology_configs = [
            a for a in label_engine_pipeline_config["nodes"] if a["name"] == "gen_dgp_semantic_map"
        ]
        if len(potential_ontology_configs) == 1:
            ontology_node = potential_ontology_configs[0]
            config = ontology_node.get("config")
            if config is not None:
                ontology_stream_name = config.get("output_ontology_path")
                if ontology_stream_name is not None:
                    return ontology_stream_name
                else:
                    raise ValueError(f"Can not find output_ontology_path in label engine config {config}.")
            else:
                raise ValueError(f"Can not find config in label engine config node {ontology_node}")
        else:
            raise ValueError(
                f"Can not resolve ontology stream name from label engine pipeline config {potential_ontology_configs}"
            )
