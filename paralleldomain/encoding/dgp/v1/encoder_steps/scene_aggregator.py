from datetime import datetime
from typing import Any, Dict, Iterable, List, Set

import pypeln

from paralleldomain.common.dgp.v1 import dataset_pb2, scene_pb2
from paralleldomain.common.dgp.v1.constants import DATETIME_FORMAT
from paralleldomain.encoding.pipeline_encoder import EncoderStep
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import relative_path, write_json_message


class DGPV1SceneAggregator(EncoderStep):
    def __init__(self, output_path: AnyPath, dataset_name: str):
        self.dataset_name = dataset_name
        self.output_path = output_path
        self.scene_paths: List[AnyPath] = list()
        self.all_annotation_types: Set[int] = set()

    def apply(self, input_stage: Iterable[Any]) -> Iterable[Any]:
        stage = input_stage
        stage = pypeln.thread.map(
            f=self.aggregate_scene, stage=stage, workers=1, maxsize=4, on_done=self.finalize_dataset
        )
        return stage

    def aggregate_scene(self, scene_info: Dict[str, Any]):
        if "scene_storage_path" in scene_info:
            scene_storage_path = scene_info["scene_storage_path"]
            available_annotation_types = scene_info["available_annotation_types"]
            self.scene_paths.append(scene_storage_path)
            self.all_annotation_types.update(available_annotation_types)
        return dict()

    def finalize_dataset(self):
        metadata_proto = dataset_pb2.DatasetMetadata(
            name=self.dataset_name,
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

        metadata_proto.available_annotation_types.extend(list(self.all_annotation_types))

        dataset_proto = dataset_pb2.SceneDataset(
            metadata=metadata_proto,  # needs refinement, currently assumes DGP->DGP
            scene_splits={
                dataset_pb2.DatasetSplit.TRAIN: scene_pb2.SceneFiles(
                    filenames=[
                        relative_path(start=self.output_path, path=scene_path).as_posix()
                        for scene_path in sorted(self.scene_paths)
                    ],
                )
            },
        )

        output_path = self.output_path / "scene_dataset.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_message(obj=dataset_proto, path=output_path)
