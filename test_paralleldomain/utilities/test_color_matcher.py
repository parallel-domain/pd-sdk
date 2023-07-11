from pathlib import Path
from typing import List

import numpy as np
import pytest

from paralleldomain.utilities.color_matcher import StreamingStats, GaussianColorDistribution, ColorMatcher


class TestStreamingStats:
    def test_mean_is_zero_when_count_zero(self):
        stats = StreamingStats()

        assert np.allclose(stats.mean, [0, 0, 0])

    def test_var_is_nan_when_count_is_below_two(self):
        stats = StreamingStats()

        assert np.all(np.isnan(stats.std))

        stats.update(np.arange(3))

        assert np.all(np.isnan(stats.std))

    def test_it_calculates_correct_mean_and_var(self):
        stats = StreamingStats()

        stats.update(np.array([[-1, 0, 10], [0, 0.1, 20], [1, 0.2, 30]]))

        assert np.allclose(stats.mean, [0, 0.1, 20])
        assert np.allclose(stats.var, [1, 0.01, 100])

    def test_it_aggregates_across_multiple_calls(self):
        stats = StreamingStats()

        stats.update(np.array([[-1, 0, 10]]))
        stats.update(np.array([[0, 0.1, 20]]))
        stats.update(np.array([[1, 0.2, 30]]))

        assert np.allclose(stats.mean, [0, 0.1, 20])
        assert np.allclose(stats.var, [1, 0.01, 100])

    def test_it_matches_numpy_for_huge_arrays(self):
        stats = StreamingStats()
        np.random.seed(0)
        data = np.random.uniform(low=-1000, high=1000, size=(1024, 3))

        stats.update(data)

        assert np.allclose(stats.mean, data.mean(axis=0))
        assert np.allclose(stats.var, data.var(axis=0, ddof=1))


class TestGaussianColorDistribution:
    def test_update_works_on_image_like_data(self):
        image = np.array(
            [
                [
                    [0, 0, 0],
                    [255, 255, 255],
                ],
                [[255, 255, 255], [0, 128, 255]],
            ],
            dtype=np.uint8,
        )
        gaussian_color_distribution = GaussianColorDistribution()

        gaussian_color_distribution.update(image)

        assert np.allclose(gaussian_color_distribution.mean, [63.7, 4.7, -17.7], atol=0.1)
        assert np.allclose(gaussian_color_distribution.var, [2257.9, 88.1, 1257.4], atol=0.1)

    def test_it_works_on_float_images(self):
        image = np.array(
            [
                [
                    [0, 0, 0],
                    [1.0, 1.0, 1.0],
                ],
                [[1.0, 1.0, 1.0], [0, 0.5, 1.0]],
            ],
            dtype=np.float32,
        )
        gaussian_color_distribution = GaussianColorDistribution()

        gaussian_color_distribution.update(image)

        assert np.allclose(gaussian_color_distribution.mean, [63.7, 4.7, -17.7], atol=0.1)
        assert np.allclose(gaussian_color_distribution.var, [2258.7, 91.19, 1265.2], atol=0.1)

    def test_it_ignores_the_alpha_channel(self):
        image = np.array(
            [
                [
                    [255, 255, 255, 128],
                ],
            ],
            dtype=np.uint8,
        )
        gaussian_color_distribution = GaussianColorDistribution()

        gaussian_color_distribution.update(image)

        assert np.allclose(gaussian_color_distribution.mean, [100, 0, 0], atol=0.1)
        assert np.all(np.isnan(gaussian_color_distribution.var))

    @pytest.mark.parametrize("use_tqdm", [True, False])
    def test_from_image_stream_works_with_generators(self, use_tqdm: bool):
        image_stream = iter(
            [
                np.array(
                    [
                        [[255, 255, 255], [0, 128, 255]],
                    ],
                    dtype=np.uint8,
                ),
                np.array(
                    [
                        [
                            [0, 0, 0],
                            [255, 255, 255],
                        ],
                    ],
                    dtype=np.uint8,
                ),
            ]
        )

        gaussian_color_distribution = GaussianColorDistribution.from_image_stream(
            image_stream=image_stream, use_tqdm=use_tqdm
        )

        assert np.allclose(gaussian_color_distribution.mean, [63.7, 4.7, -17.7], atol=0.1)
        assert np.allclose(gaussian_color_distribution.var, [2257.9, 88.1, 1257.4], atol=0.1)

    def test_save_and_load(self, tmp_path: Path):
        image = np.array(
            [
                [
                    [0, 0, 0],
                    [255, 255, 255],
                ],
                [[255, 255, 255], [0, 128, 255]],
            ],
            dtype=np.uint8,
        )
        save_path = str(tmp_path / "distribution.json")
        gaussian_color_distribution = GaussianColorDistribution()
        gaussian_color_distribution.update(image)

        gaussian_color_distribution.save_to_json(save_path)
        loaded_distribution = GaussianColorDistribution.from_json(save_path)

        assert np.allclose(gaussian_color_distribution.mean, loaded_distribution.mean)
        assert np.allclose(gaussian_color_distribution.var, loaded_distribution.var)


class TestHueTransform:
    def test_from_distribution_creates_identity_transformation_matrix_if_source_equal_target(self):
        distribution = GaussianColorDistribution()
        distribution._mean = np.array([100, 100, 100])
        distribution._count = 3
        distribution._M2 = 2 * np.array([2, 2, 2])

        color_transform = ColorMatcher.from_distributions(source=distribution, target=distribution)

        assert np.allclose(color_transform.transformation_matrix, np.identity(4))

    def test_from_distribution_creates_correct_color_transformation_matrix(self):
        source_distribution = GaussianColorDistribution()
        source_distribution._mean = np.array([100, 100, 100])
        source_distribution._count = 3
        source_distribution._M2 = 2 * np.array([4, 4, 4])  # std dev of 2
        target_distribution = GaussianColorDistribution()
        target_distribution._mean = np.array([10, 20, 100])
        target_distribution._count = 3
        target_distribution._M2 = 2 * np.array([1, 4, 0.01])  # std dev of 1, 2, 0.1
        expected_transform = [[0.5, 0, 0, -40], [0, 1, 0, -80], [0, 0, 0.05, 95], [0, 0, 0, 1]]

        color_matcher = ColorMatcher.from_distributions(source=source_distribution, target=target_distribution)

        assert np.allclose(color_matcher.transformation_matrix, expected_transform)

    def test_matmul_applies_identity_transform_to_image(self):
        color_matcher = ColorMatcher(transformation_matrix=np.identity(4).astype(np.float32))
        colors = np.arange(60).astype(np.uint8).reshape((5, 4, 3))

        result = color_matcher @ colors

        # There is some rounding error when converting rgb 2 lab and back. We could reduce that by working with floats,
        # but we might break existing transforms
        assert np.allclose(result, colors, atol=2)

    def test_matmul_works_with_float_images(self):
        color_matcher = ColorMatcher(transformation_matrix=np.identity(4).astype(np.float32))
        colors = np.arange(60).reshape((5, 4, 3)).astype(np.float32) / 100.0

        result = color_matcher @ colors

        # There is some rounding error when converting rgb 2 lab and back. We could reduce that by working with floats,
        # but we might break existing transforms
        assert np.allclose(result, colors, atol=2)

    def test_matmul_applies_transform_in_lab_space_and_keeps_alpha(self):
        color_matcher = ColorMatcher(
            transformation_matrix=np.array(
                [[1, 0, 0, 10], [0, 0.1, 0, 1], [0, 0, 0.2, 2], [0, 0, 0, 1]], dtype=np.float32
            )
        )
        colors = np.arange(8).astype(np.uint8).reshape((1, 2, 4))
        expected = np.array([[30, 27, 25, 3], [32, 29, 27, 7]])

        result = color_matcher @ colors

        assert np.allclose(result, expected)

    @pytest.mark.parametrize(
        "shape",
        [
            [1, 2],
            [1, 2, 2],
            [
                2,
            ],
            [2, 4, 5],
        ],
    )
    def test_matmul_raises_for_invalid_shape(self, shape: List[int]):
        color_matcher = ColorMatcher(transformation_matrix=np.identity(4))
        colors = np.ones(shape, np.uint8)

        with pytest.raises(ValueError):
            color_matcher @ colors

    def test_save_and_load(self, tmp_path: Path):
        save_path = str(tmp_path / "distribution.json")
        color_matcher = ColorMatcher(
            transformation_matrix=np.array(
                [[1, 0, 0, 10], [0, 0.1, 0, 1], [0, 0, 0.2, 2], [0, 0, 0, 1]], dtype=np.float32
            )
        )

        color_matcher.save_to_json(save_path)
        loaded_transform = ColorMatcher.from_json(save_path)

        assert np.allclose(color_matcher.transformation_matrix, loaded_transform.transformation_matrix)
