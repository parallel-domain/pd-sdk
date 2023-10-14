import pickle
from tempfile import TemporaryDirectory
from typing import Dict, Generator
from uuid import uuid4

from paralleldomain.encoding.dgp.v1.format.common import CAMERA_DATA_FOLDER, CUSTOM_FORMAT_KEY, LIDAR_DATA_FOLDER
from paralleldomain.encoding.pipeline_encoder import ScenePipelineItem
from paralleldomain.utilities.any_path import AnyPath


class DataAggregationMixin:
    def __init__(self):
        self.aggregation_temp_folders: Dict[str, AnyPath] = dict()

    @staticmethod
    def store_item_for_aggregation(data: ScenePipelineItem, path: AnyPath):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_aggregated_item(path: AnyPath) -> ScenePipelineItem:
        with path.open("rb") as handle:
            return pickle.load(handle)

    def store_data_for_aggregation(self, pipeline_item: ScenePipelineItem, output_folder: AnyPath):
        keep_data = pipeline_item.custom_data[CUSTOM_FORMAT_KEY]
        pipeline_item.custom_data = dict()
        pipeline_item.custom_data[CUSTOM_FORMAT_KEY] = keep_data
        data_to_store = pipeline_item
        # data_to_store = pipeline_item.custom_data[CUSTOM_FORMAT_KEY]
        file_name = f"{uuid4()}.pickle"
        if pipeline_item.camera_frame is not None:
            storage_folder = (
                self.get_scene_tmp_aggregation_folder(scene_name=pipeline_item.scene_name, output_folder=output_folder)
                / CAMERA_DATA_FOLDER
            )
            # data_to_store["frame_id"] = pipeline_item.camera_frame.frame_id
        elif pipeline_item.lidar_frame is not None:
            storage_folder = (
                self.get_scene_tmp_aggregation_folder(scene_name=pipeline_item.scene_name, output_folder=output_folder)
                / LIDAR_DATA_FOLDER
            )
            # data_to_store["frame_id"] = pipeline_item.lidar_frame.frame_id
        else:
            return
        self.store_item_for_aggregation(data=data_to_store, path=storage_folder / file_name)

    def load_data_for_aggregation(self, folder_path: AnyPath) -> Generator[ScenePipelineItem, None, None]:
        for path in folder_path.iterdir():
            if str(path).endswith("pickle"):
                yield self.load_aggregated_item(path=path)

    def load_camera_data_for_aggregation(
        self, scene_name: str, output_folder: AnyPath
    ) -> Generator[ScenePipelineItem, None, None]:
        folder_path = (
            self.get_scene_tmp_aggregation_folder(scene_name=scene_name, output_folder=output_folder)
            / CAMERA_DATA_FOLDER
        )
        if folder_path.exists():
            yield from self.load_data_for_aggregation(folder_path=folder_path)

    def load_lidar_data_for_aggregation(
        self, scene_name: str, output_folder: AnyPath
    ) -> Generator[ScenePipelineItem, None, None]:
        folder_path = (
            self.get_scene_tmp_aggregation_folder(scene_name=scene_name, output_folder=output_folder)
            / LIDAR_DATA_FOLDER
        )
        if folder_path.exists():
            yield from self.load_data_for_aggregation(folder_path=folder_path)

    def get_scene_tmp_aggregation_folder(self, scene_name: str, output_folder: AnyPath) -> AnyPath:
        if scene_name not in self.aggregation_temp_folders:
            self.aggregation_temp_folders[scene_name] = output_folder / "tmp" / scene_name
        return self.aggregation_temp_folders[scene_name]

    def clean_up_scene_tmp_aggregation_folder(self, scene_name: str):
        folder_path = None
        if scene_name in self.aggregation_temp_folders:
            folder_path = self.aggregation_temp_folders.pop(scene_name)
            for path in list(reversed(list(folder_path.rglob("*")))):
                path.rm(missing_ok=True)
            if folder_path.exists():
                folder_path.rmdir()

        if folder_path is not None and folder_path.parent.exists() and len(list(folder_path.parent.iterdir())) == 0:
            folder_path.parent.rmdir()
