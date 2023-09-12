"""
Contains classes for statistical matching of image color statistics.
This is one of the simplest and least artifact causing image style transfer techniques. We found that properly
calibrated hue transform improves performance across several tasks.
"""

from dataclasses import dataclass
from typing import Union, Generator, Optional, Iterator

import cv2
import numpy as np
import ujson
from tqdm import tqdm

from paralleldomain import Dataset
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image

__all__ = ["GaussianColorDistribution", "ColorMatcher"]


class StreamingStats:
    def __init__(self):
        self._count = 0
        self._mean = np.zeros(3, dtype=float)
        self._M2 = np.zeros(3, dtype=float)

    def update(self, x: np.ndarray):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # Welford's online algorithm for single pass variance calculation
        n = x.shape[0]
        self._count += n
        delta = x - self._mean
        self._mean += np.sum(delta, axis=0) / self._count
        delta2 = x - self._mean
        self._M2 += np.sum(delta * delta2, axis=0)

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def var(self) -> np.ndarray:
        if self._count < 2:
            return np.full(3, np.nan)
        else:
            return self._M2 / (self._count - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)


class GaussianColorDistribution(StreamingStats):
    """
    Tracks per channel mean and variance in LAB space
    """

    @property
    def to_distribution(self) -> np.ndarray:
        """
        Transformation that can be used to transform a color value from a unit distribution into this distribution

        Returns:
            4x4 transformation matrix
        """
        matrix = np.identity(4).astype(np.float32)
        matrix[:3, :3] *= self.std
        matrix[:3, 3] = self.mean
        return matrix

    @property
    def to_unit_distribution(self) -> np.ndarray:
        """
        Transformation that can be used to transform a color value from this distribution into a unit distribution

        Returns:
            4x4 transformation matrix
        """
        return np.linalg.inv(self.to_distribution)

    def save_to_json(self, path: Union[str, AnyPath]) -> None:
        path = AnyPath(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(count=self._count, mean=self._mean.tolist(), m2=self._M2.tolist())
        with path.open("w") as f:
            ujson.dump(data, f)

    def update(self, x: np.ndarray) -> None:
        """
        adds the image colors to the color statistics

        Args:
            x: image in rgb
        """
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255
        elif x.dtype != np.float32:
            raise ValueError("Only np.uint8 and np.float32 images are supported")
        if x.shape[-1] == 4:
            x = x[..., :3]
        img = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
        img = img.reshape((-1, 3))
        super().update(x=img)

    @staticmethod
    def from_json(path: Union[str, AnyPath]) -> "GaussianColorDistribution":
        path = AnyPath(path)
        with path.open("r") as f:
            data = ujson.load(f)
        distribution = GaussianColorDistribution()
        distribution._count = data["count"]
        distribution._mean = np.array(data["mean"])
        distribution._M2 = np.array(data["m2"])
        return distribution

    @staticmethod
    def from_image_stream(image_stream: Iterator[np.ndarray], use_tqdm: bool = True) -> "GaussianColorDistribution":
        """
        Calculates the statistics of all images in the given image stream

        Args:
            image_stream: Iterator over rgb images
            use_tqdm: shows a progress bar if true

        Returns:
            The color distribution object
        """
        stats = GaussianColorDistribution()
        if use_tqdm:
            image_stream = tqdm(image_stream, unit="Images")
        for image in image_stream:
            stats.update(image)
        return stats

    @staticmethod
    def from_folder(image_folder: Union[str, AnyPath], use_tqdm: bool = True) -> "GaussianColorDistribution":
        """
        Calculates color statistics of all images in a folder. Note that this only works on flat folders right now

        Args:
            image_folder: folder with images
            use_tqdm: shows a progress bar if true

        Returns:
            The color distribution object
        """
        image_gen = (
            read_image(path=path) for path in AnyPath(image_folder).iterdir() if path.suffix in [".jpg", ".png"]
        )

        return GaussianColorDistribution.from_image_stream(image_stream=image_gen, use_tqdm=use_tqdm)

    @staticmethod
    def from_dataset(dataset: Dataset, use_tqdm: bool = True, max_samples: int = -1) -> "GaussianColorDistribution":
        """
        Calculates color statistics of images in a dataset

        Args:
            dataset: the dataset instance
            use_tqdm: shows a progress bar if true
            max_samples: number of images that are used to calculate images. It takes all images if set to -1

        Returns:
            The color distribution object
        """

        def image_gen() -> Generator[np.ndarray, None, None]:
            cnt = 0
            for sensor_frame, _, _ in dataset.sensor_frame_pipeline(shuffle=True, concurrent=True, only_cameras=True):
                if max_samples > 0 and cnt > max_samples:
                    break

                if isinstance(sensor_frame, CameraSensorFrame):
                    yield sensor_frame.image.rgb
                    cnt += 1

        return GaussianColorDistribution.from_image_stream(image_stream=image_gen(), use_tqdm=use_tqdm)

    @staticmethod
    def from_dataset_path(
        dataset_path: Union[str, AnyPath],
        dataset_format: str = "dgp",
        settings: Optional[DecoderSettings] = None,
        use_tqdm: bool = True,
        max_samples: int = -1,
        **decoder_kwargs,
    ) -> "GaussianColorDistribution":
        dataset = decode_dataset(
            dataset_path=dataset_path, dataset_format=dataset_format, settings=settings, **decoder_kwargs
        )

        return GaussianColorDistribution.from_dataset(dataset=dataset, use_tqdm=use_tqdm, max_samples=max_samples)


@dataclass
class ColorMatcher:
    """
    Precomputed transform that can be applied to images to match the color distributions of two datasets.
    """

    transformation_matrix: np.ndarray

    def save_to_json(self, path: Union[str, AnyPath]):
        path = AnyPath(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            ujson.dump(self.transformation_matrix.tolist(), f)

    @staticmethod
    def from_json(path: Union[str, AnyPath]) -> "ColorMatcher":
        path = AnyPath(path)
        with path.open("r") as f:
            data = ujson.load(f)
        return ColorMatcher(transformation_matrix=np.array(data))

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        """
        Applies the color matching to a rgb/rgba image

        Args:
            other: the image as numpy array

        Returns:
            the color transformed image
        """
        if not isinstance(other, np.ndarray):
            raise ValueError(f"Invalid value {other}! Has to be an numpy array of shape MxNx3 or  MxNx4!")

        original_dtype = other.dtype
        original_shape = other.shape

        transform_input = other
        if other.dtype == np.uint8:
            transform_input = transform_input.astype(np.float32) / 255
        elif other.dtype != np.float32:
            raise ValueError()

        if len(original_shape) != 3 or original_shape[-1] not in [3, 4]:
            raise ValueError(f"Invalid shape {original_shape}! Has to be an numpy array of shape MxNx3 or  MxNx4!")

        transform_input = cv2.cvtColor(transform_input, cv2.COLOR_RGB2LAB)

        flattened_input = transform_input.reshape(-1, 3).T

        homogeneous = np.concatenate([flattened_input, np.ones((1, flattened_input.shape[1]), dtype=np.float32)])
        projected = self.transformation_matrix @ homogeneous
        projected = projected[:3, :]
        projected = projected.T

        projected = projected.reshape(transform_input.shape)
        projected_rgb = cv2.cvtColor(projected, cv2.COLOR_LAB2RGB)

        projected_rgb = np.clip(projected_rgb, 0.0, 1.0)
        if original_dtype == np.uint8:
            projected_rgb = projected_rgb * 255
        projected_rgb = projected_rgb.astype(original_dtype)

        if original_shape[-1] == 4:
            projected_rgb = np.concatenate([projected_rgb, other[..., 3:4]], axis=-1)

        return projected_rgb

    @staticmethod
    def from_distributions(source: GaussianColorDistribution, target: GaussianColorDistribution) -> "ColorMatcher":
        """
        Calculates the color transform from source to target distribution
        """
        transformation_matrix = target.to_distribution @ source.to_unit_distribution
        return ColorMatcher(transformation_matrix=transformation_matrix)
