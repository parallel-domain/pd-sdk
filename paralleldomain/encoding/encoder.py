import argparse
import concurrent.futures
import os
import uuid
from typing import Any, List, Optional

from paralleldomain import Dataset
from paralleldomain.decoding.dgp.decoder import DGPDecoder
from paralleldomain.utilities.any_path import AnyPath


class SceneEncoder:
    def __init__(self, dataset: Dataset, scene_name: str, output_path: AnyPath):
        self._dataset = dataset
        self._scene_name = scene_name
        self._output_path = output_path

    def run(self) -> Any:
        print(f"Successfully encoded {self._scene_name}")
        return str(uuid.uuid4())

    @staticmethod
    def encode(dataset: Dataset, scene_name: str, output_path: AnyPath):
        return SceneEncoder(dataset=dataset, scene_name=scene_name, output_path=output_path).run()


class DatasetEncoder:
    scene_encoder: SceneEncoder = SceneEncoder

    def __init__(
        self,
        input_path: str,
        output_path: str,
        scene_names: Optional[List[str]] = None,
        scene_start: Optional[int] = None,
        scene_stop: Optional[int] = None,
        n_parallel: Optional[int] = 1,
    ):
        self._input_path = AnyPath(input_path)
        self._output_path = AnyPath(output_path)
        self._n_processes = min(max(n_parallel, 1), os.cpu_count())

        self._load_dataset()

        if scene_names is not None:
            for sn in scene_names:
                if sn not in self._dataset.scene_names:
                    raise KeyError(f"{sn} could not be found in dataset {self._dataset.name}")
            self._scene_names = scene_names
        else:
            scene_slice = slice(scene_start, scene_stop)
            self._scene_names = self._dataset.scene_names[scene_slice]

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._n_processes) as scene_executor:
            for scene_name, scene_encoder_result in zip(
                self._scene_names,
                scene_executor.map(
                    lambda sn: type(self).scene_encoder.encode(self._dataset, sn, self._output_path), self._scene_names
                ),
            ):
                print(f"{scene_name}: {scene_encoder_result}")

    def _load_dataset(self) -> None:
        """Simple dataset loader - naively assumes input is DGP format"""
        decoder = DGPDecoder(dataset_path=self._input_path)
        self._dataset = Dataset.from_decoder(decoder=decoder)


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
        "--n_parallel",
        type=int,
        help="Define how many scenes should be processed in parallel",
        required=False,
        default=1,
    )

    args = parser.parse_args()

    DatasetEncoder(
        input_path=args.input,
        output_path=args.output,
        scene_names=args.scene_names,
        scene_start=args.scene_start,
        scene_stop=args.scene_stop,
        n_parallel=args.n_parallel,
    ).run()
