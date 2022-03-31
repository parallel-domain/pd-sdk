from datetime import datetime
from typing import Optional, Union

import numpy as np

from paralleldomain.common.dgp.v1 import dataset_pb2, scene_pb2
from paralleldomain.common.dgp.v1.constants import DATETIME_FORMAT, DirectoryName
from paralleldomain.encoding.dgp.v1.format.aggregation import DataAggregationMixin
from paralleldomain.encoding.dgp.v1.format.common import (
    CUSTOM_FORMAT_KEY,
    ENCODED_SCENE_AGGREGATION_FOLDER_NAME,
    SCENE_DATA_KEY,
    SENSOR_DATA_KEY,
    CommonDGPV1FormatMixin,
)
from paralleldomain.encoding.pipeline_encoder import PipelineItem, ScenePipelineItem
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import relative_path, write_json_message


class DatasetDGPV1Mixin(CommonDGPV1FormatMixin, DataAggregationMixin):
    def save_aggregated_dataset(
        self, pipeline_item: ScenePipelineItem, dataset_output_path: AnyPath, target_dataset_name: Optional[str]
    ):
        scene_paths = list()
        all_annotation_types = set()
        for scene_pipeline_items in self.load_data_for_aggregation(
            folder_path=dataset_output_path / ENCODED_SCENE_AGGREGATION_FOLDER_NAME
        ):
            scene_info = scene_pipeline_items.custom_data[CUSTOM_FORMAT_KEY][SCENE_DATA_KEY]
            scene_storage_path = scene_info["scene_storage_path"]
            available_annotation_types = scene_info["available_annotation_types"]
            scene_paths.append(scene_storage_path)
            all_annotation_types.update(available_annotation_types)

        dataset_name = target_dataset_name if target_dataset_name is not None else pipeline_item.dataset.name
        metadata_proto = dataset_pb2.DatasetMetadata(
            name=dataset_name,
            version="1.0",
            creation_date=datetime.utcnow().strftime(DATETIME_FORMAT),
            creator="PD",
            bucket_path=None,
            raw_path=None,
            description="",
            origin=dataset_pb2.DatasetMetadata.DatasetOrigin.INTERNAL,
            available_annotation_types=[],
            statistics=None,
            frame_per_second=0.0,
            metadata={},
        )

        metadata_proto.available_annotation_types.extend(list(all_annotation_types))

        dataset_proto = dataset_pb2.SceneDataset(
            metadata=metadata_proto,  # needs refinement, currently assumes DGP->DGP
            scene_splits={
                dataset_pb2.DatasetSplit.TRAIN: scene_pb2.SceneFiles(
                    filenames=[
                        relative_path(start=dataset_output_path, path=scene_path).as_posix()
                        for scene_path in sorted(scene_paths)
                    ],
                )
            },
        )

        output_path = dataset_output_path / "scene_dataset.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_message(obj=dataset_proto, path=output_path)

        aggregation_folder = dataset_output_path / ENCODED_SCENE_AGGREGATION_FOLDER_NAME
        for path in aggregation_folder.iterdir():
            path.rm(missing_ok=True)
        aggregation_folder.rmdir()
