import argparse
import logging
from contextlib import suppress
from datetime import datetime
from typing import Dict, List, Optional, Type, Union

from paralleldomain.common.dgp.v1 import dataset_pb2, scene_pb2
from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP_INV, DATETIME_FORMAT
from paralleldomain.decoding.decoder import DatasetDecoder
from paralleldomain.decoding.dgp.decoder import DGPDatasetDecoder
from paralleldomain.encoding.dgp.v1.scene import DGPSceneEncoder
from paralleldomain.encoding.encoder import DatasetEncoder, SceneEncoder
from paralleldomain.model.annotation import Annotation, AnnotationType
from paralleldomain.model.dataset import Dataset
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import write_json_message
from paralleldomain.utilities.logging import setup_loggers

logger = logging.getLogger(__name__)


class DGPDatasetEncoder(DatasetEncoder):
    def __init__(
        self,
        dataset: Dataset,
        output_path: str,
        dataset_name: str = None,
        scene_names: Optional[List[str]] = None,
        scene_start: Optional[int] = None,
        scene_stop: Optional[int] = None,
    ) -> None:
        super().__init__(
            dataset=dataset,
            output_path=output_path,
            scene_names=scene_names,
            set_start=scene_start,
            set_stop=scene_stop,
        )
        self._dataset_name: str = dataset_name

        self._scene_encoder: Type[SceneEncoder] = DGPSceneEncoder
        # Adapt if should be limited to a set of cameras, or empty list for no cameras
        self._camera_names: Union[List[str], None] = None
        # Adapt if should be limited to a set of lidars, or empty list for no lidars
        self._lidar_names: Union[List[str], None] = None
        # Adapt if should be limited to a set of frames, or empty list for no frames
        self._frame_ids: Optional[List[str]] = None
        # Adapt if should be limited to a set of annotation types, or empty list for no annotations
        self._annotation_types: Union[List[AnnotationType], None] = None

    def _encode_dataset_json(self, scene_files: Dict[str, AnyPath]) -> AnyPath:
        metadata_proto = dataset_pb2.DatasetMetadata(
            name=(self._dataset_name if self._dataset_name else self._dataset.name),
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

        if self._annotation_types:
            metadata_proto.available_annotation_types.extend(
                [
                    int(ANNOTATION_TYPE_MAP_INV[a_type])
                    for a_type in self._annotation_types
                    if a_type is not Annotation  # equiv: not implemented, yet!
                ]
            )
        else:
            metadata_proto.available_annotation_types.extend(
                [
                    int(ANNOTATION_TYPE_MAP_INV[a_type])
                    for a_type in self._dataset.available_annotation_types
                    if a_type is not Annotation  # equiv: not implemented, yet!
                ]
            )

        dataset_proto = dataset_pb2.SceneDataset(
            metadata=metadata_proto,  # needs refinement, currently assumes DGP->DGP
            scene_splits={
                dataset_pb2.DatasetSplit.TRAIN: scene_pb2.SceneFiles(
                    filenames=[
                        self._relative_path(scene_files[scene_key]).as_posix()
                        for scene_key in sorted(scene_files.keys())
                    ],
                )
            },
        )

        output_path = self._output_path / "scene_dataset.json"
        return write_json_message(obj=dataset_proto, path=output_path)

    def encode_dataset(self) -> AnyPath:
        scene_files = {}
        for scene_name in self._scene_names:
            # try:
            scene_files[scene_name] = self._call_scene_encoder(scene_name=scene_name)
            # except Exception:
            #     continue

        return self._encode_dataset_json(scene_files=scene_files)

    @staticmethod
    def from_dataset(
        dataset: Dataset,
        output_path: str,
        dataset_name: str = None,
        scene_names: Optional[List[str]] = None,
        scene_start: Optional[int] = None,
        scene_stop: Optional[int] = None,
    ) -> "DGPDatasetEncoder":
        return DGPDatasetEncoder(
            dataset=dataset,
            output_path=output_path,
            dataset_name=dataset_name,
            scene_names=scene_names,
            scene_start=scene_start,
            scene_stop=scene_stop,
        )

    @staticmethod
    def from_path(
        input_path: str,
        output_path: str,
        dataset_name: str = None,
        scene_names: Optional[List[str]] = None,
        scene_start: Optional[int] = None,
        scene_stop: Optional[int] = None,
    ) -> "DGPDatasetEncoder":
        # Todo detect decoder type from path content
        decoder = DGPDatasetDecoder(dataset_path=input_path)
        return DGPDatasetEncoder.from_decoder(
            decoder=decoder,
            output_path=output_path,
            dataset_name=dataset_name,
            scene_names=scene_names,
            scene_start=scene_start,
            scene_stop=scene_stop,
        )

    @staticmethod
    def from_decoder(
        decoder: DatasetDecoder,
        output_path: str,
        dataset_name: str = None,
        scene_names: Optional[List[str]] = None,
        scene_start: Optional[int] = None,
        scene_stop: Optional[int] = None,
    ) -> "DGPDatasetEncoder":
        return DGPDatasetEncoder.from_dataset(
            dataset=decoder.get_dataset(),
            output_path=output_path,
            dataset_name=dataset_name,
            scene_names=scene_names,
            scene_start=scene_start,
            scene_stop=scene_stop,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a data encoders")
    parser.add_argument("-i", "--input", type=str, help="A local or cloud path to a DGP dataset", required=True)
    parser.add_argument("-o", "--output", type=str, help="A local or cloud path for the encoded dataset", required=True)
    parser.add_argument(
        "--scene_names",
        nargs="*",
        type=str,
        help="""Define one or multiple specific scenes to be processed.
                When provided, overwrites any scene_start and scene_stop arguments""",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--scene_start",
        type=int,
        help="An integer defining the start index for scene processing",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--scene_stop",
        type=int,
        help="An integer defining the stop index for scene processing",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Define the output name of the encoded dataset. Leave empty to reuse input dataset name",
        required=False,
        default=None,
    )

    args = parser.parse_args()

    setup_loggers([__name__, "fsio"], log_level=logging.DEBUG)

    DGPDatasetEncoder.from_path(
        input_path=args.input,
        output_path=args.output,
        dataset_name=args.dataset_name,
        scene_names=args.scene_names,
        scene_start=args.scene_start,
        scene_stop=args.scene_stop,
    ).encode_dataset()
