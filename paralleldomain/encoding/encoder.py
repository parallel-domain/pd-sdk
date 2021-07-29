import argparse
import logging
import os
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
from typing import Any, Generator, List, Optional, Union

import numpy as np

from paralleldomain import Dataset, Scene
from paralleldomain.decoding.dgp.decoder import DGPDecoder
from paralleldomain.encoding.utils.log import setup_loggers
from paralleldomain.model.sensor import SensorFrame
from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger(__name__)


class ObjectFilter:
    @staticmethod
    def filter_pre_transform(objects: Union[Generator, List]) -> Union[Generator, List]:
        return (o for o in objects)

    @staticmethod
    def transform(objects: Union[Generator, List]) -> Union[Generator, List]:
        return (o for o in objects)

    @staticmethod
    def filter_post_transform(objects: Union[Generator, List]) -> Union[Generator, List]:
        return (o for o in objects)

    @classmethod
    def run(cls, objects: List) -> List:
        _pre_filtered = cls.filter_pre_transform(objects=objects)
        _transformed = cls.transform(objects=_pre_filtered)
        _post_filtered = cls.filter_post_transform(objects=_transformed)
        return list(_post_filtered)


class MaskFilter:
    @staticmethod
    def transform(mask: np.ndarray) -> np.ndarray:
        return mask

    @classmethod
    def run(cls, mask: np.ndarray) -> np.ndarray:
        _transformed = cls.transform(mask)
        return _transformed


class SceneEncoder:
    _logger: logging.Logger = None

    def __init__(self, dataset: Dataset, scene_name: str, output_path: AnyPath):
        self._dataset: Dataset = dataset
        self._scene_name: str = scene_name
        self._output_path: AnyPath = output_path
        self._scene: Scene = dataset.get_scene(scene_name)
        self._task_pool = ThreadPool(processes=max(int(os.cpu_count() * 0.75), 1))

        self._prepare_output_directories()

    def _run_async(self, func, *args, **kwargs):
        return self._task_pool.apply_async(func, args=args, kwds=dict(**kwargs))

    def _prepare_output_directories(self) -> None:
        self._output_path.mkdir(exist_ok=True, parents=True)

    def _encode_camera_frame(self, camera_frame: SensorFrame):
        ...

    def _encode_lidar_frame(self, lidar_frame: SensorFrame):
        ...

    def _encode_camera(self, camera_name: str):
        with ThreadPoolExecutor(max_workers=10) as camera_frame_executor:
            for frame_id, camera_frame_encoder_result in zip(
                self._scene.frame_ids,
                camera_frame_executor.map(
                    lambda fid: self._encode_camera_frame(self._scene.get_frame(fid).get_sensor(camera_name)),
                    self._scene.frame_ids,
                ),
            ):
                self.logger().info(f"{camera_name} - {frame_id}: {camera_frame_encoder_result}")

    def _encode_cameras(self):
        with ThreadPoolExecutor(max_workers=4) as camera_executor:
            for camera_name, camera_encoder_result in zip(
                self._scene.camera_names, camera_executor.map(self._encode_camera, self._scene.camera_names)
            ):
                self.logger().info(f"{camera_name}: {camera_encoder_result}")

    def _encode_lidar(self, lidar_name: str):
        with ThreadPoolExecutor(max_workers=10) as lidar_frame_executor:
            for frame_id, lidar_frame_encoder_result in zip(
                self._scene.frame_ids,
                lidar_frame_executor.map(
                    lambda fid: self._encode_lidar_frame(self._scene.get_frame(fid).get_sensor(lidar_name)),
                    self._scene.frame_ids,
                ),
            ):
                self.logger().info(f"{lidar_name} - {frame_id}: {lidar_frame_encoder_result}")

    def _encode_lidars(self):
        with ThreadPoolExecutor(max_workers=4) as lidar_executor:
            for lidar_name, lidar_encoder_result in zip(
                self._scene.lidar_names, lidar_executor.map(self._encode_lidar, self._scene.lidar_names)
            ):
                self.logger().info(f"{lidar_name}: {lidar_encoder_result}")

    def _encode_sensors(self):
        self._encode_cameras()
        self._encode_lidars()

    def _run_encoding(self) -> Any:
        self._encode_sensors()
        self.logger().info(f"Successfully encoded {self._scene_name}")
        return str(uuid.uuid4())

    def run(self) -> Any:
        encoding_result = self._run_encoding()
        self._task_pool.close()
        self._task_pool.join()

        return encoding_result

    @classmethod
    def encode(cls, dataset: Dataset, scene_name: str, output_path: AnyPath) -> Any:
        return cls(dataset=dataset, scene_name=scene_name, output_path=output_path).run()

    @classmethod
    def logger(cls):
        if cls._logger is None:
            cls._logger = logging.getLogger(name=cls.__name__)
            setup_loggers([cls.__name__], log_level=logging.DEBUG)
        return cls._logger


class DatasetEncoder:
    scene_encoder: SceneEncoder = SceneEncoder
    _logger: logging.Logger = None

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

    def _call_scene_encoder(self, scene_name: str) -> Any:
        return type(self).scene_encoder.encode(self._dataset, scene_name, self._output_path / scene_name)

    def run(self):
        with ThreadPoolExecutor(max_workers=self._n_processes) as scene_executor:
            for scene_name, scene_encoder_result in zip(
                self._scene_names,
                scene_executor.map(
                    self._call_scene_encoder,
                    self._scene_names,
                ),
            ):
                self.logger().info(f"{scene_name}: {scene_encoder_result}")

    def _load_dataset(self) -> None:
        """Simple dataset loader - naively assumes input is DGP format"""
        decoder = DGPDecoder(dataset_path=self._input_path)
        self._dataset = Dataset.from_decoder(decoder=decoder)

    @classmethod
    def logger(cls):
        if cls._logger is None:
            cls._logger = logging.getLogger(name=cls.__name__)
            setup_loggers([cls.__name__], log_level=logging.DEBUG)
        return cls._logger


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
