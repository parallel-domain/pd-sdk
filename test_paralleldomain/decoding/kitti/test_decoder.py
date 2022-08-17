import os

import numpy as np
import pytest

from paralleldomain import Dataset
from paralleldomain.decoding.kitti.decoder import KITTIDatasetDecoder
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.sensor import CameraSensor, CameraSensorFrame
from paralleldomain.model.unordered_scene import UnorderedScene
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image
from test_paralleldomain.decoding.constants import KITTI_DATASET_PATH_ENV


@pytest.fixture
def kitti_dataset_path() -> str:
    if KITTI_DATASET_PATH_ENV in os.environ:
        return os.environ[KITTI_DATASET_PATH_ENV]
    else:
        pytest.skip()
        # return "s3://pd-internal-ml/flow/KITTI2015"


@pytest.fixture
def kitti_train_dataset(kitti_dataset_path: str) -> Dataset:
    decoder = KITTIDatasetDecoder(dataset_path=kitti_dataset_path)
    dataset = decoder.get_dataset()
    return dataset


@pytest.fixture()
def kitti_dataset_train_scene(kitti_train_dataset: Dataset) -> UnorderedScene:
    scene_names = kitti_train_dataset.unordered_scene_names
    scene = kitti_train_dataset.get_unordered_scene(scene_name=scene_names[0])
    return scene


def test_can_load_scene(kitti_dataset_train_scene: UnorderedScene):
    assert kitti_dataset_train_scene is not None


def test_knows_all_frames(kitti_dataset_train_scene: UnorderedScene):
    assert (
        len(kitti_dataset_train_scene.frame_ids) == 200
    )  # numer of frames mentioned here http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow


def test_decode_train_scene_names(kitti_train_dataset: Dataset):
    assert len(kitti_train_dataset.scene_names) == 0
    scene_names = kitti_train_dataset.unordered_scene_names
    assert 1 == len(scene_names)


def test_decode_camera_image(kitti_dataset_train_scene: UnorderedScene):
    for i in range(5):
        camera_frame = next(kitti_dataset_train_scene.camera_frames)
        assert camera_frame is not None
        assert isinstance(camera_frame, CameraSensorFrame)
        image = camera_frame.image
        assert image is not None
        rgb = image.rgb
        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (375, 1242, 3)
        assert rgb.shape[0] == image.height
        assert rgb.shape[1] == image.width
        assert rgb.shape[2] == image.channels


def test_decode_optical_flow(kitti_dataset_train_scene: UnorderedScene):
    for i in range(5):
        camera_frame = next(kitti_dataset_train_scene.camera_frames)
        assert camera_frame is not None
        assert isinstance(camera_frame, CameraSensorFrame)
        flow = camera_frame.get_annotations(annotation_type=AnnotationTypes.OpticalFlow)
        assert flow is not None
        vectors = flow.vectors
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (375, 1242, 2)
        assert np.all(np.logical_and(flow.vectors.max() < 1024, flow.vectors.min() >= 0))


def test_decode_camera_image_next(kitti_dataset_train_scene: UnorderedScene):
    dataset_path = kitti_dataset_train_scene.metadata["dataset_path"]
    image_folder = kitti_dataset_train_scene.metadata["image_folder"]
    for i in range(5):
        camera_frame = next(kitti_dataset_train_scene.camera_frames)
        image = camera_frame.image
        fid = camera_frame.frame_id
        fid_next = fid_next = fid[:-5] + "1.png"
        next_image_path = AnyPath(dataset_path) / image_folder / fid_next
        next_image = read_image(next_image_path)
        assert next_image is not None
        assert isinstance(next_image, np.ndarray)
        assert next_image.shape == (375, 1242, 3)
        assert next_image.shape[0] == image.height
        assert next_image.shape[1] == image.width
        assert next_image.shape[2] == image.channels
