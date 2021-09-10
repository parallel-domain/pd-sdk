import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Type, Union

from paralleldomain.common.dgp.v0.constants import ANNOTATION_TYPE_MAP_INV, DATETIME_FORMAT
from paralleldomain.common.dgp.v0.dtos import DatasetDTO, DatasetMetaDTO, DatasetSceneSplitDTO
from paralleldomain.decoding.decoder import DatasetDecoder
from paralleldomain.decoding.dgp.decoder import DGPDatasetDecoder
from paralleldomain.encoding.dgp.scene import DGPSceneEncoder
from paralleldomain.encoding.encoder import DatasetEncoder, SceneEncoder
from paralleldomain.model.annotation import Annotation, AnnotationType
from paralleldomain.model.dataset import Dataset
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath
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
        n_parallel: Optional[int] = 1,
    ) -> None:
        super().__init__(
            dataset=dataset,
            output_path=output_path,
            scene_names=scene_names,
            set_start=scene_start,
            set_stop=scene_stop,
            n_parallel=n_parallel,
        )
        self._dataset_name: str = dataset_name

        self._scene_encoder: Type[SceneEncoder] = DGPSceneEncoder
        # Adapt if should be limited to a set of cameras, or empty list for no cameras
        self._camera_names: Union[List[str], None] = None
        # Adapt if should be limited to a set of lidars, or empty list for no lidars
        self._lidar_names: Union[List[str], None] = None
        # Adapt if should be limited to a set of annotation types, or empty list for no annotations
        self._annotation_types: Union[List[AnnotationType], None] = None

    def _encode_dataset_json(self, scene_files: Dict[str, AnyPath]) -> AnyPath:
        metadata_dto = DatasetMetaDTO(**self._dataset.metadata.custom_attributes)
        metadata_dto.name = self._dataset_name if self._dataset_name else self._dataset.name
        metadata_dto.creation_date = datetime.utcnow().strftime(DATETIME_FORMAT)
        if self._annotation_types:
            metadata_dto.available_annotation_types = [
                int(ANNOTATION_TYPE_MAP_INV[a_type])
                for a_type in self._annotation_types
                if a_type is not Annotation  # equiv: not implemented, yet!
            ]
        else:
            metadata_dto.available_annotation_types = [
                int(ANNOTATION_TYPE_MAP_INV[a_type])
                for a_type in self._dataset.available_annotation_types
                if a_type is not Annotation  # equiv: not implemented, yet!
            ]

        ds_dto = DatasetDTO(
            metadata=metadata_dto,  # needs refinement, currently assumes DGP->DGP
            scene_splits={
                "0": DatasetSceneSplitDTO(
                    filenames=[
                        self._relative_path(scene_files[scene_key]).as_posix()
                        for scene_key in sorted(scene_files.keys())
                    ],
                )
            },
        )

        output_path = self._output_path / "scene_dataset.json"
        return fsio.write_json(obj=ds_dto.to_dict(), path=output_path)

    def encode_dataset(self) -> AnyPath:
        with ThreadPoolExecutor(max_workers=self._n_parallel) as scene_executor:
            scene_files = dict(
                zip(
                    self._scene_names,
                    scene_executor.map(
                        self._call_scene_encoder,
                        self._scene_names,
                    ),
                )
            )

        return self._encode_dataset_json(scene_files=scene_files)

    @staticmethod
    def from_dataset(
        dataset: Dataset,
        output_path: str,
        dataset_name: str = None,
        scene_names: Optional[List[str]] = None,
        scene_start: Optional[int] = None,
        scene_stop: Optional[int] = None,
        n_parallel: Optional[int] = 1,
    ):
        return DGPDatasetEncoder(
            dataset=dataset,
            output_path=output_path,
            dataset_name=dataset_name,
            scene_names=scene_names,
            scene_start=scene_start,
            scene_stop=scene_stop,
            n_parallel=n_parallel,
        )

    @staticmethod
    def from_path(
        input_path: str,
        output_path: str,
        dataset_name: str = None,
        scene_names: Optional[List[str]] = None,
        scene_start: Optional[int] = None,
        scene_stop: Optional[int] = None,
        n_parallel: Optional[int] = 1,
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
            n_parallel=n_parallel,
        )

    @staticmethod
    def from_decoder(
        decoder: DatasetDecoder,
        output_path: str,
        dataset_name: str = None,
        scene_names: Optional[List[str]] = None,
        scene_start: Optional[int] = None,
        scene_stop: Optional[int] = None,
        n_parallel: Optional[int] = 1,
    ) -> "DGPDatasetEncoder":
        return DGPDatasetEncoder.from_dataset(
            dataset=decoder.get_dataset(),
            output_path=output_path,
            dataset_name=dataset_name,
            scene_names=scene_names,
            scene_start=scene_start,
            scene_stop=scene_stop,
            n_parallel=n_parallel,
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

    parser.add_argument(
        "--n_parallel",
        type=int,
        help="Define how many scenes should be processed in parallel",
        required=False,
        default=1,
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
        n_parallel=args.n_parallel,
    ).encode_dataset()
