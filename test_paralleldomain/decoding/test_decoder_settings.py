import os
from typing import Any, Dict

import pytest

from paralleldomain import Dataset
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.utilities.lazy_load_cache import LAZY_LOAD_CACHE
from test_paralleldomain.decoding.constants import (
    CITYSCAPES_DATASET_PATH_ENV,
    DGP_DATASET_PATH_ENV,
    KITTI_FLOW_DATASET_PATH_ENV,
    NUIMAGES_DATASET_PATH_ENV,
)


@pytest.fixture(
    params=[
        ("dgp", DGP_DATASET_PATH_ENV, dict()),
        ("cityscapes", CITYSCAPES_DATASET_PATH_ENV, dict(splits=["test"])),
        ("nuimages", NUIMAGES_DATASET_PATH_ENV, dict(split="v1.0-mini")),
        ("kitti-flow", KITTI_FLOW_DATASET_PATH_ENV, dict(split_name="training")),
    ]
)
def dataset_params(request) -> Dict[str, Any]:
    dataset_format, path_env, kwargs = request.param

    if path_env not in os.environ:
        pytest.skip()
    dataset_path = os.environ[path_env]

    return dict(dataset_path=dataset_path, dataset_format=dataset_format, **kwargs)


@pytest.fixture(params=[True, False])
def with_caching(request) -> bool:
    return request.param


@pytest.fixture()
def dataset(dataset_params: Dict[str, Any], with_caching: bool) -> Dataset:
    settings = DecoderSettings(
        cache_images=with_caching, cache_point_clouds=with_caching, cache_annotations=with_caching
    )
    dataset = decode_dataset(settings=settings, **dataset_params)
    return dataset


def test_decoder_cache_default_setting_is_all_false():
    settings = DecoderSettings()
    assert not settings.cache_point_clouds
    assert not settings.cache_annotations
    assert not settings.cache_images


class TestImageCaching:
    def test_cache_size_on_image_load(self, dataset: Dataset, with_caching: bool):
        LAZY_LOAD_CACHE.clear()
        scene = dataset.get_unordered_scene(scene_name=dataset.unordered_scene_names[0])
        if len(scene.camera_names) == 0:
            pytest.skip()

        camera = scene.get_camera_sensor(camera_name=scene.camera_names[0])

        camera_frame = camera.get_frame(frame_id=list(camera.frame_ids)[-1])

        image = camera_frame.image

        num_keys_before_load = len(LAZY_LOAD_CACHE.keys())
        rgb = image.rgb
        assert rgb is not None
        if with_caching:
            assert num_keys_before_load < len(LAZY_LOAD_CACHE.keys())
        else:
            assert num_keys_before_load == len(LAZY_LOAD_CACHE.keys())


class TestCloudCaching:
    def test_cache_size_on_cloud_load(self, dataset: Dataset, with_caching: bool):
        LAZY_LOAD_CACHE.clear()
        scene = dataset.get_unordered_scene(scene_name=dataset.unordered_scene_names[0])
        if len(scene.lidar_names) == 0:
            pytest.skip()

        lidar = scene.get_lidar_sensor(lidar_name=scene.lidar_names[0])
        lidar_frame = lidar.get_frame(frame_id=list(lidar.frame_ids)[-1])
        cloud = lidar_frame.point_cloud

        num_keys_before_load = len(LAZY_LOAD_CACHE.keys())
        xyz = cloud.xyz
        assert xyz is not None
        if with_caching:
            assert num_keys_before_load < len(LAZY_LOAD_CACHE.keys())
        else:
            assert num_keys_before_load == len(LAZY_LOAD_CACHE.keys())


class TestAnnotationCaching:
    def test_cache_size_on_annotation_load(self, dataset: Dataset, with_caching: bool):
        LAZY_LOAD_CACHE.clear()
        scene = dataset.get_unordered_scene(scene_name=dataset.unordered_scene_names[0])
        if len(scene.camera_names) == 0:
            pytest.skip()

        camera = scene.get_camera_sensor(camera_name=scene.camera_names[0])

        camera_frame = camera.get_frame(frame_id=list(camera.frame_ids)[-1])
        annotation_types = camera_frame.available_annotation_types
        if AnnotationTypes.PointCaches in annotation_types:
            # Point Caches load 3d boxes, which are then part of the cache, breaking the test case
            annotation_types.remove(AnnotationTypes.PointCaches)
            annotation_types.append(AnnotationTypes.PointCaches)

        for annotype in annotation_types:
            keys_before_load = set(LAZY_LOAD_CACHE.keys())
            try:
                anno = camera_frame.get_annotations(annotation_type=annotype)
                assert anno is not None
                if with_caching:
                    assert len(keys_before_load) < len(LAZY_LOAD_CACHE.keys())
                else:
                    new_keys = set(LAZY_LOAD_CACHE.keys()) - keys_before_load
                    assert len(new_keys) == 0 or "file_path" in next(iter(new_keys))
            except NotImplementedError:
                pass

    def test_cache_size_on_lidar_annotation_load(self, dataset: Dataset, with_caching: bool):
        LAZY_LOAD_CACHE.clear()
        scene = dataset.get_unordered_scene(scene_name=dataset.unordered_scene_names[0])
        if len(scene.lidar_names) == 0:
            pytest.skip()
        lidar = scene.get_lidar_sensor(lidar_name=scene.lidar_names[0])
        lidar_frame = lidar.get_frame(frame_id=list(lidar.frame_ids)[-1])
        annotation_types = lidar_frame.available_annotation_types
        if AnnotationTypes.PointCaches in annotation_types:
            # Point Caches load 3d boxes, which are then part of the cache, breaking the test case
            annotation_types.remove(AnnotationTypes.PointCaches)
            annotation_types.append(AnnotationTypes.PointCaches)

        for annotype in annotation_types:
            num_keys_before_load = len(LAZY_LOAD_CACHE.keys())
            try:
                anno = lidar_frame.get_annotations(annotation_type=annotype)
                assert anno is not None
                if with_caching:
                    assert num_keys_before_load < len(LAZY_LOAD_CACHE.keys())
                else:
                    assert num_keys_before_load == len(LAZY_LOAD_CACHE.keys())
            except NotImplementedError:
                pass
