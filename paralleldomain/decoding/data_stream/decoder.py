from datetime import datetime
from typing import Union, Optional, List, Dict, Any, Set, overload

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.data_stream.data_accessor import (
    DataStreamDataAccessor,
    StoredDataStreamDataAccessor,
    LabelEngineDataStreamDataAccessor,
)
from paralleldomain.decoding.data_stream.frame_decoder import DataStreamFrameDecoder
from paralleldomain.decoding.data_stream.sensor_decoder import DataStreamCameraSensorDecoder
from paralleldomain.decoding.decoder import DatasetDecoder, SceneDecoder, TDateTime
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_decoder import RadarSensorDecoder, LidarSensorDecoder, CameraSensorDecoder
from paralleldomain.model.annotation import AnnotationType, AnnotationIdentifier, AnnotationTypes
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.type_aliases import SceneName, FrameId, SensorName
from paralleldomain.utilities.any_path import AnyPath
from pd.data_lab import LabeledStateReference

StreamType = int


class DataStreamDatasetDecoder(DatasetDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        settings: Optional[DecoderSettings] = None,
        camera_image_stream_name: str = "rgb",
        available_annotation_identifiers: Optional[List[AnnotationIdentifier]] = None,
    ):
        self._init_kwargs = dict(
            dataset_path=dataset_path,
            settings=settings,
        )
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        self._camera_image_stream_name = camera_image_stream_name
        self._available_annotation_identifiers = available_annotation_identifiers

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
            camera_image_stream_name=self._camera_image_stream_name,
            available_annotation_identifiers=self._available_annotation_identifiers,
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
        scene_path: AnyPath,
        available_annotation_identifiers: Optional[List[AnnotationIdentifier]] = None,
        camera_image_stream_name: str = "rgb",
    ):
        ...

    @overload
    def __init__(
        self,
        *,
        dataset_name: str,
        settings: DecoderSettings,
        state_reference: LabeledStateReference,
        available_annotation_identifiers: List[AnnotationIdentifier],
        camera_image_stream_name: str = "rgb",
    ):
        ...

    def __init__(
        self,
        *,
        dataset_name: str,
        settings: DecoderSettings,
        state_reference: Optional[LabeledStateReference] = None,
        scene_path: Optional[AnyPath] = None,
        available_annotation_identifiers: Optional[List[AnnotationIdentifier]] = None,
        camera_image_stream_name: str = "rgb",
    ):
        super().__init__(dataset_name=dataset_name, settings=settings)
        self._scene_path = scene_path
        if scene_path is not None:
            if available_annotation_identifiers is None:
                available_annotation_identifiers = [
                    AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D, name="bounding_box_2d_xyz"),
                    AnnotationIdentifier(
                        annotation_type=AnnotationTypes.InstanceSegmentation2D, name="instance_mask_xyz"
                    ),
                    AnnotationIdentifier(
                        annotation_type=AnnotationTypes.SemanticSegmentation2D, name="semantic_mask_xyz"
                    ),
                ]
            self._data_accessor: DataStreamDataAccessor = StoredDataStreamDataAccessor(
                scene_path=scene_path,
                camera_image_stream_name=camera_image_stream_name,
                potentially_available_annotation_identifiers=available_annotation_identifiers,
            )
        else:
            if available_annotation_identifiers is None:
                raise ValueError("available_annotation_identifiers is required when using label engine!")
            self._data_accessor = LabelEngineDataStreamDataAccessor(
                labeled_state_reference=state_reference,
                camera_image_stream_name=camera_image_stream_name,
                available_annotation_identifiers=available_annotation_identifiers,
            )

    def update_labeled_state_reference(self, labeled_state_reference: LabeledStateReference) -> None:
        if not isinstance(self._data_accessor, LabelEngineDataStreamDataAccessor):
            raise ValueError("Can only update labeled state reference on LabelEngineDataStreamDataAccessor")
        self._data_accessor.update_labeled_state_reference(labeled_state_reference)

    def _decode_set_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
        return self._data_accessor.get_scene_metadata()

    def _decode_set_description(self, scene_name: SceneName) -> str:
        return ""

    def _decode_frame_id_set(self, scene_name: SceneName) -> Set[FrameId]:
        return self._data_accessor.get_frame_ids()

    def _decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        return self._data_accessor.sensor_names

    def _decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        return self._data_accessor.camera_names

    def _decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        return self._data_accessor.lidar_names

    def _decode_radar_names(self, scene_name: SceneName) -> List[SensorName]:
        return self._data_accessor.radar_names

    def _decode_class_maps(self, scene_name: SceneName) -> Dict[AnnotationType, ClassMap]:
        return dict()

    def _create_camera_sensor_decoder(
        self, scene_name: SceneName, camera_name: SensorName, dataset_name: str
    ) -> CameraSensorDecoder[TDateTime]:
        return DataStreamCameraSensorDecoder(
            dataset_name=self.dataset_name,
            scene_name=scene_name,
            settings=self.settings,
            data_accessor=self._data_accessor,
        )

    def _create_lidar_sensor_decoder(
        self, scene_name: SceneName, lidar_name: SensorName, dataset_name: str
    ) -> LidarSensorDecoder[TDateTime]:
        raise NotImplementedError("Lidar decoding not implemented")

    def _create_radar_sensor_decoder(
        self, scene_name: SceneName, radar_name: SensorName, dataset_name: str
    ) -> RadarSensorDecoder[TDateTime]:
        raise NotImplementedError("Radar decoding not implemented")

    def _create_frame_decoder(
        self, scene_name: SceneName, frame_id: FrameId, dataset_name: str
    ) -> FrameDecoder[TDateTime]:
        return DataStreamFrameDecoder(
            dataset_name=dataset_name, scene_name=scene_name, settings=self.settings, data_accessor=self._data_accessor
        )

    def _decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, TDateTime]:
        return self._data_accessor.get_frame_id_to_date_time_map()

    def _decode_available_annotation_identifiers(self, scene_name: SceneName) -> List[AnnotationIdentifier]:
        return self._data_accessor.available_annotation_identifiers
