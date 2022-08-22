import argparse
import concurrent
import itertools
import logging
import os
import uuid
from abc import abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from tempfile import TemporaryDirectory
from typing import Any, Callable, Generator, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import time

from paralleldomain import Dataset
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.sensor import SensorFrame
from paralleldomain.model.type_aliases import SceneName, SensorName
from paralleldomain.model.unordered_scene import UnorderedScene
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import relative_path
from paralleldomain.utilities.os import cpu_count

logger = logging.getLogger(__name__)

_thread_pool_size = max(int(os.environ.get("ENCODER_THREAD_POOL_MAX_SIZE", cpu_count() * 4)), 4)


class EncoderThreadPool(ThreadPoolExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = self._max_workers

    def map_async(self, fn: Callable[[Any], Any], iterable: Iterable[Any]) -> List[Future]:
        return [self.submit(fn, i) for i in iterable]


ENCODING_THREAD_POOL = EncoderThreadPool(_thread_pool_size)


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
        scene_name: SceneName,
        output_path: AnyPath,
        camera_names: Optional[List[str]] = None,
        lidar_names: Optional[List[str]] = None,
        frame_ids: Optional[List[str]] = None,
        annotation_types: Optional[List[AnnotationType]] = None,
    ):
        self._dataset: Dataset = dataset
        self._scene_name: SceneName = scene_name
        self._output_path: AnyPath = output_path
        self._unordered_scene: UnorderedScene = dataset.get_unordered_scene(scene_name=scene_name)

        self._camera_names: Optional[List[str]] = (
            self._unordered_scene.camera_names if camera_names is None else camera_names
        )
        self._lidar_names: Optional[List[str]] = (
            self._unordered_scene.lidar_names if lidar_names is None else lidar_names
        )
        self._frame_ids: Optional[List[str]] = self._unordered_scene.frame_ids if frame_ids is None else frame_ids
        self._annotation_types: Optional[List[AnnotationType]] = (
            self._unordered_scene.available_annotation_types if annotation_types is None else annotation_types
        )

        self._prepare_output_directories()

    @property
    def _sensor_names(self) -> List[str]:
        return self._camera_names + self._lidar_names

    def _relative_path(self, path: AnyPath) -> AnyPath:
        return relative_path(path, self._output_path)

    def _run_async(self, func: Callable[[Any], Any], *args, **kwargs) -> Future:
        return ENCODING_THREAD_POOL.submit(func, *args, **kwargs)

    def _prepare_output_directories(self) -> None:
        if not self._output_path.is_cloud_path:
            self._output_path.mkdir(exist_ok=True, parents=True)

    @abstractmethod
    def _encode_camera_frame(self, camera_name: str, camera_frame: SensorFrame):
        ...

    @abstractmethod
    def _encode_lidar_frame(self, frame_id: str, lidar_frame: SensorFrame):
        ...

    def _encode_cameras(self) -> Any:
        return [self._encode_camera(camera_name=c).result() for c in self._camera_names]

    def _encode_camera(self, camera_name: SensorName) -> Future:
        frame_ids = self._frame_ids
        camera_encoding_futures = {
            ENCODING_THREAD_POOL.submit(
                lambda fid: self._encode_camera_frame(
                    camera_name=camera_name,
                    camera_frame=self._unordered_scene.get_frame(frame_id=fid).get_sensor(sensor_name=camera_name),
                ),
                frame_id,
            )
            for frame_id in frame_ids
        }
        return ENCODING_THREAD_POOL.submit(lambda: concurrent.futures.wait(camera_encoding_futures))

    def _encode_lidar(self, lidar_name: SensorName) -> Future:
        frame_ids = self._frame_ids
        lidar_encoding_futures = {
            ENCODING_THREAD_POOL.submit(
                lambda fid: self._encode_lidar_frame(
                    frame_id=frame_id,
                    lidar_frame=self._unordered_scene.get_frame(frame_id=fid).get_sensor(sensor_name=lidar_name),
                ),
                frame_id,
            )
            for frame_id in frame_ids
        }
        return ENCODING_THREAD_POOL.submit(lambda: concurrent.futures.wait(lidar_encoding_futures))

    def _encode_lidars(self) -> Any:
        return [self._encode_lidar(lidar_name=ln).result() for ln in self._lidar_names]

    def _encode_sensors(self):
        self._encode_cameras()
        self._encode_lidars()

    def _run_encoding(self) -> Any:
        self._encode_sensors()
        logger.info(f"Successfully encoded {self._scene_name}")
        return str(uuid.uuid4())

    def encode_scene(self) -> Any:
        encoding_result = self._run_encoding()
        return encoding_result


class DatasetEncoder:
    def __init__(
        self,
        dataset: Dataset,
        output_path: str,
        scene_names: Optional[List[str]] = None,
        set_start: Optional[int] = None,
        set_stop: Optional[int] = None,
        sync_after_scene_encoded: bool = True,
    ):
        self._dataset = dataset
        self._output_path = AnyPath(output_path)
        self._sync_after_scene_encoded = sync_after_scene_encoded

        # Adapt to use specific SceneEncoder type
        self._scene_encoder: Type[SceneEncoder] = SceneEncoder
        # Adapt if should be limited to a set of cameras, or empty list for no cameras
        self._camera_names: Union[List[str], None] = None
        # Adapt if should be limited to a set of lidars, or empty list for no lidars
        self._lidar_names: Union[List[str], None] = None
        # Adapt if should be limited to a set of frames, or empty list for no frames
        self._frame_ids: Optional[List[str]] = None
        # Adapt if should be limited to a set of annotation types, or empty list for no annotations
        self._annotation_types: Union[List[AnnotationType], None] = None

        if scene_names is not None:
            for sn in scene_names:
                if sn not in self._dataset.unordered_scene_names:
                    raise KeyError(f"{sn} could not be found in dataset {self._dataset.name}")
            self._scene_names = scene_names
        else:
            set_slice = slice(set_start, set_stop)
            self._scene_names = self._dataset.unordered_scene_names[set_slice]

    def _call_scene_encoder(self, scene_name: str) -> AnyPath:
        remote_output_dir = self._output_path / scene_name
        if self._sync_after_scene_encoded:
            temp_dir = TemporaryDirectory()
            output_dir = AnyPath(temp_dir.name)
        else:
            output_dir = remote_output_dir

        encoder = self._scene_encoder(
            dataset=self._dataset,
            scene_name=scene_name,
            output_path=output_dir,
            camera_names=self._camera_names,
            lidar_names=self._lidar_names,
            frame_ids=self._frame_ids,
            annotation_types=self._annotation_types,
        )
        result = encoder.encode_scene()
        # TODO: properly wait for all jobs to finish instead of using time.sleep
        time.sleep(60)  # give threadpool chance to finish before copying / deleting
        if self._sync_after_scene_encoded:
            if remote_output_dir.is_cloud_path:
                output_dir.sync(target=remote_output_dir)
            else:
                output_dir.copytree(target=remote_output_dir)
            try:
                temp_dir.cleanup()
            except OSError as e:
                logger.warning(f"Could not delete temp directory. Ignoring and continuing. Error: {e}")
        return remote_output_dir / result.parts[-1]

    def _relative_path(self, path: AnyPath) -> AnyPath:
        return relative_path(path, self._output_path)

    def encode_dataset(self):
        for scene_name, scene_encoder_result in zip(
            self._scene_names,
            ENCODING_THREAD_POOL.map(
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
        set_start: Optional[int] = None,
        set_stop: Optional[int] = None,
    ) -> "DatasetEncoder":
        return cls(
            dataset=dataset,
            output_path=output_path,
            scene_names=scene_names,
            set_start=set_start,
            set_stop=set_stop,
        )

    @classmethod
    def from_path(
        cls,
        input_path: str,
        output_path: str,
        scene_names: Optional[List[str]] = None,
        scene_start: Optional[int] = None,
        scene_stop: Optional[int] = None,
    ) -> "DatasetEncoder":
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

    args = parser.parse_args()

    DatasetEncoder.from_path(
        input_path=args.input,
        output_path=args.output,
        scene_names=args.scene_names,
        scene_start=args.scene_start,
        scene_stop=args.scene_stop,
    ).encode_dataset()
