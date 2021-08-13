import argparse
import logging
import os
import uuid
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
from typing import Any, Generator, List, Optional, Type, Union
from urllib.parse import urlparse

import numpy as np

from paralleldomain import Dataset, Scene
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.sensor import SensorFrame
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import relative_path

logger = logging.getLogger(__name__)


class ObjectTransformer:
    @staticmethod
    def _filter_pre_transform(objects: Union[Generator, List]) -> Union[Generator, List]:
        return (o for o in objects)

    @staticmethod
    def _transform(objects: Union[Generator, List]) -> Union[Generator, List]:
        return (o for o in objects)

    @staticmethod
    def _filter_post_transform(objects: Union[Generator, List]) -> Union[Generator, List]:
        return (o for o in objects)

    @classmethod
    def transform(cls, objects: List) -> List:
        _pre_filtered = cls._filter_pre_transform(objects=objects)
        _transformed = cls._transform(objects=_pre_filtered)
        _post_filtered = cls._filter_post_transform(objects=_transformed)
        return list(_post_filtered)


class MaskTransformer:
    @staticmethod
    def _transform(mask: np.ndarray) -> np.ndarray:
        return mask

    @classmethod
    def transform(cls, mask: np.ndarray) -> np.ndarray:
        _transformed = cls._transform(mask)
        return _transformed


class SceneEncoder:
    def __init__(
        self,
        dataset: Dataset,
        scene_name: str,
        output_path: AnyPath,
        camera_names: Optional[Union[List[str], None]] = None,
        lidar_names: Optional[Union[List[str], None]] = None,
        annotation_types: Optional[Union[List[AnnotationType], None]] = None,
    ):
        self._dataset: Dataset = dataset
        self._scene_name: str = scene_name
        self._output_path: AnyPath = output_path
        self._scene: Scene = dataset.get_scene(scene_name)
        self._task_pool: ThreadPool = ThreadPool(processes=max(int(os.cpu_count() * 0.75), 1))

        self._camera_names: Union[List[str], None] = self._scene.camera_names if camera_names is None else camera_names
        self._lidar_names: Union[List[str], None] = self._scene.lidar_names if lidar_names is None else lidar_names
        self._annotation_types: Union[List[AnnotationType], None] = (
            self._scene.available_annotation_types if annotation_types is None else annotation_types
        )

        self._prepare_output_directories()

    @property
    def _sensor_names(self) -> List[str]:
        return self._camera_names + self._lidar_names

    def _relative_path(self, path: AnyPath) -> AnyPath:
        return relative_path(path, self._output_path)

    def _run_async(self, func, *args, **kwargs):
        return self._task_pool.apply_async(func, args=args, kwds=dict(**kwargs))

    def _prepare_output_directories(self) -> None:
        if not urlparse(str(self._output_path)).scheme:
            self._output_path.mkdir(exist_ok=True, parents=True)

    @abstractmethod
    def _encode_camera_frame(self, camera_frame: SensorFrame):
        ...

    @abstractmethod
    def _encode_lidar_frame(self, lidar_frame: SensorFrame):
        ...

    def _encode_camera(self, camera_name: str):
        with ThreadPoolExecutor(max_workers=10) as camera_frame_executor:
            camera_frame_executor.map(
                lambda fid: self._encode_camera_frame(self._scene.get_frame(fid).get_sensor(camera_name)),
                self._scene.frame_ids,
            ),

    def _encode_cameras(self):
        with ThreadPoolExecutor(max_workers=4) as camera_executor:
            for camera_name, camera_encoder_result in zip(
                self._camera_names, camera_executor.map(self._encode_camera, self._camera_names)
            ):
                logger.info(f"{camera_name}: {camera_encoder_result}")

    def _encode_lidar(self, lidar_name: str):
        with ThreadPoolExecutor(max_workers=10) as lidar_frame_executor:
            for frame_id, lidar_frame_encoder_result in zip(
                self._scene.frame_ids,
                lidar_frame_executor.map(
                    lambda fid: self._encode_lidar_frame(self._scene.get_frame(fid).get_sensor(lidar_name)),
                    self._scene.frame_ids,
                ),
            ):
                logger.info(f"{lidar_name} - {frame_id}: {lidar_frame_encoder_result}")

    def _encode_lidars(self):
        with ThreadPoolExecutor(max_workers=4) as lidar_executor:
            for lidar_name, lidar_encoder_result in zip(
                self._lidar_names, lidar_executor.map(self._encode_lidar, self._lidar_names)
            ):
                logger.info(f"{lidar_name}: {lidar_encoder_result}")

    def _encode_sensors(self):
        self._encode_cameras()
        self._encode_lidars()

    def _run_encoding(self) -> Any:
        self._encode_sensors()
        logger.info(f"Successfully encoded {self._scene_name}")
        return str(uuid.uuid4())

    def encode_scene(self) -> Any:
        encoding_result = self._run_encoding()
        self._task_pool.close()
        self._task_pool.join()

        return encoding_result


class DatasetEncoder:
    def __init__(
        self,
        dataset: Dataset,
        output_path: str,
        scene_names: Optional[List[str]] = None,
        scene_start: Optional[int] = None,
        scene_stop: Optional[int] = None,
        n_parallel: Optional[int] = 1,
    ):
        self._dataset = dataset
        self._output_path = AnyPath(output_path)
        self._n_parallel = min(max(n_parallel, 1), os.cpu_count())

        # Adapt to use specific SceneEncoder type
        self._scene_encoder: Type[SceneEncoder] = SceneEncoder
        # Adapt if should be limited to a set of cameras, or empty list for no cameras
        self._camera_names: Union[List[str], None] = None
        # Adapt if should be limited to a set of lidars, or empty list for no lidars
        self._lidar_names: Union[List[str], None] = None
        # Adapt if should be limited to a set of annotation types, or empty list for no annotations
        self._annotation_types: Union[List[AnnotationType], None] = None

        if scene_names is not None:
            for sn in scene_names:
                if sn not in self._dataset.scene_names:
                    raise KeyError(f"{sn} could not be found in dataset {self._dataset.name}")
            self._scene_names = scene_names
        else:
            scene_slice = slice(scene_start, scene_stop)
            self._scene_names = self._dataset.scene_names[scene_slice]

    def _call_scene_encoder(self, scene_name: str) -> Any:
        encoder = self._scene_encoder(
            dataset=self._dataset,
            scene_name=scene_name,
            output_path=self._output_path / scene_name,
            camera_names=self._camera_names,
            lidar_names=self._lidar_names,
            annotation_types=self._annotation_types,
        )
        return encoder.encode_scene()

    def _relative_path(self, path: AnyPath) -> AnyPath:
        return relative_path(path, self._output_path)

    def encode_dataset(self):
        with ThreadPoolExecutor(max_workers=self._n_parallel) as scene_executor:
            for scene_name, scene_encoder_result in zip(
                self._scene_names,
                scene_executor.map(
                    self._call_scene_encoder,
                    self._scene_names,
                ),
            ):
                logger.info(f"{scene_name}: {scene_encoder_result}")

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        output_path: str,
        scene_names: Optional[List[str]] = None,
        scene_start: Optional[int] = None,
        scene_stop: Optional[int] = None,
        n_parallel: Optional[int] = 1,
    ):
        return cls(
            dataset=dataset,
            output_path=output_path,
            scene_names=scene_names,
            scene_start=scene_start,
            scene_stop=scene_stop,
            n_parallel=n_parallel,
        )

    @classmethod
    def from_path(
        cls,
        input_path: str,
        output_path: str,
        scene_names: Optional[List[str]] = None,
        scene_start: Optional[int] = None,
        scene_stop: Optional[int] = None,
        n_parallel: Optional[int] = 1,
    ):
        raise NotImplementedError("An Encoder needs to override this method with a fitting Decoder")


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

    DatasetEncoder.from_path(
        input_path=args.input,
        output_path=args.output,
        scene_names=args.scene_names,
        scene_start=args.scene_start,
        scene_stop=args.scene_stop,
        n_parallel=args.n_parallel,
    )._transform()
