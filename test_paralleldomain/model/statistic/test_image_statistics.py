from datetime import datetime

import numpy as np

from paralleldomain import Scene
from paralleldomain.decoding.in_memory.scene_decoder import InMemorySceneDecoder
from paralleldomain.decoding.in_memory.sensor_frame_decoder import InMemoryCameraFrameDecoder
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.sensor import CameraSensorFrame, SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.statistics import ImageStatistics


def test_all_red():
    image_statistics = ImageStatistics()

    image = np.zeros([512, 512, 3], dtype=np.uint8)
    image[..., 0] = 255

    decoder = InMemoryCameraFrameDecoder(
        dataset_name="test",
        scene_name="test",
        sensor_name="test_sensor",
        frame_id="1",
        extrinsic=SensorExtrinsic(),
        sensor_pose=SensorPose(),
        annotations={},
        class_maps={},
        intrinsic=SensorIntrinsic(),
        rgba=image,
        image_dimensions=image.shape,
        distortion_lookup=None,
        metadata={},
        date_time=datetime.now(),
    )
    sensor_frame = CameraSensorFrame(decoder=decoder)
    scene_decoder = InMemorySceneDecoder(
        scene_name="test",
    )
    scene = Scene(decoder=scene_decoder)
    image_statistics.update(scene=scene, sensor_frame=sensor_frame)

    hist_red = np.zeros(256, dtype=np.uint32)
    hist_red[-1] = 512 * 512
    hist_other = np.zeros(256, dtype=np.uint32)
    hist_other[0] = 512 * 512

    assert np.equal(image_statistics._recorder["histogram_red"], hist_red).all()
    assert np.equal(image_statistics._recorder["histogram_green"], hist_other).all()
    assert np.equal(image_statistics._recorder["histogram_blue"], hist_other).all()


def test_all_green():
    image_statistics = ImageStatistics()

    image = np.zeros([512, 512, 3], dtype=np.uint8)
    image[..., 1] = 255

    decoder = InMemoryCameraFrameDecoder(
        dataset_name="test",
        scene_name="test",
        sensor_name="test_sensor",
        frame_id="1",
        extrinsic=SensorExtrinsic(),
        sensor_pose=SensorPose(),
        annotations={},
        class_maps={},
        intrinsic=SensorIntrinsic(),
        rgba=image,
        image_dimensions=image.shape,
        distortion_lookup=None,
        metadata={},
        date_time=datetime.now(),
    )
    sensor_frame = CameraSensorFrame(decoder=decoder)
    scene_decoder = InMemorySceneDecoder(
        scene_name="test",
    )
    scene = Scene(decoder=scene_decoder)
    image_statistics.update(scene=scene, sensor_frame=sensor_frame)

    hist_green = np.zeros(256, dtype=np.uint32)
    hist_green[-1] = 512 * 512
    hist_other = np.zeros(256, dtype=np.uint32)
    hist_other[0] = 512 * 512

    assert np.equal(image_statistics._recorder["histogram_red"], hist_other).all()
    assert np.equal(image_statistics._recorder["histogram_green"], hist_green).all()
    assert np.equal(image_statistics._recorder["histogram_blue"], hist_other).all()


def test_all_blue():
    image_statistics = ImageStatistics()

    image = np.zeros([512, 512, 3], dtype=np.uint8)
    image[..., 2] = 255

    decoder = InMemoryCameraFrameDecoder(
        dataset_name="test",
        scene_name="test",
        sensor_name="test_sensor",
        frame_id="1",
        extrinsic=SensorExtrinsic(),
        sensor_pose=SensorPose(),
        annotations=[],
        class_maps={},
        intrinsic=SensorIntrinsic(),
        rgba=image,
        image_dimensions=image.shape,
        distortion_lookup=None,
        metadata={},
        date_time=datetime.now(),
    )
    sensor_frame = CameraSensorFrame(decoder=decoder)
    scene_decoder = InMemorySceneDecoder(
        scene_name="test",
    )
    scene = Scene(decoder=scene_decoder)
    image_statistics.update(scene=scene, sensor_frame=sensor_frame)

    hist_blue = np.zeros(256, dtype=np.uint32)
    hist_blue[-1] = 512 * 512
    hist_other = np.zeros(256, dtype=np.uint32)
    hist_other[0] = 512 * 512

    assert np.equal(image_statistics._recorder["histogram_red"], hist_other).all()
    assert np.equal(image_statistics._recorder["histogram_green"], hist_other).all()
    assert np.equal(image_statistics._recorder["histogram_blue"], hist_blue).all()


def test_black():
    image_statistics = ImageStatistics()

    image = np.zeros([512, 512, 3], dtype=np.uint8)
    decoder = InMemoryCameraFrameDecoder(
        dataset_name="test",
        scene_name="test",
        sensor_name="test_sensor",
        frame_id="1",
        extrinsic=SensorExtrinsic(),
        sensor_pose=SensorPose(),
        annotations={},
        class_maps={},
        intrinsic=SensorIntrinsic(),
        rgba=image,
        image_dimensions=image.shape,
        distortion_lookup=None,
        metadata={},
        date_time=datetime.now(),
    )
    sensor_frame = CameraSensorFrame(decoder=decoder)
    scene_decoder = InMemorySceneDecoder(
        scene_name="test",
    )
    scene = Scene(decoder=scene_decoder)
    image_statistics.update(scene=scene, sensor_frame=sensor_frame)

    hist_other = np.zeros(256)
    hist_other[0] = 512 * 512

    assert np.equal(image_statistics._recorder["histogram_red"], hist_other).all()
    assert np.equal(image_statistics._recorder["histogram_green"], hist_other).all()
    assert np.equal(image_statistics._recorder["histogram_blue"], hist_other).all()


def test_multiple_updates():
    image_statistics = ImageStatistics()

    image = np.zeros([512, 512, 3], dtype=np.uint8)
    image[..., 0] = 255

    decoder = InMemoryCameraFrameDecoder(
        dataset_name="test",
        scene_name="test",
        sensor_name="test_sensor",
        frame_id="1",
        extrinsic=SensorExtrinsic(),
        sensor_pose=SensorPose(),
        annotations={},
        class_maps={},
        intrinsic=SensorIntrinsic(),
        rgba=image,
        image_dimensions=image.shape,
        distortion_lookup=None,
        metadata={},
        date_time=datetime.now(),
    )
    sensor_frame = CameraSensorFrame(decoder=decoder)
    scene_decoder = InMemorySceneDecoder(
        scene_name="test",
    )
    scene = Scene(decoder=scene_decoder)
    image_statistics.update(scene=scene, sensor_frame=sensor_frame)
    image_statistics.update(scene=scene, sensor_frame=sensor_frame)
    image_statistics.update(scene=scene, sensor_frame=sensor_frame)
    image_statistics.update(scene=scene, sensor_frame=sensor_frame)

    hist_red = np.zeros(256, dtype=np.uint32)
    hist_red[-1] = 512 * 512 * 4
    hist_other = np.zeros(256, dtype=np.uint32)
    hist_other[0] = 512 * 512 * 4

    assert np.equal(image_statistics._recorder["histogram_red"], hist_red).all()
    assert np.equal(image_statistics._recorder["histogram_green"], hist_other).all()
    assert np.equal(image_statistics._recorder["histogram_blue"], hist_other).all()


def test_range():
    image_statistics = ImageStatistics()

    image = np.zeros([512, 512, 3], dtype=np.uint8)
    image[..., 0] = np.floor(np.arange(0, 256, 256 / (512 * 512))).reshape([512, 512]).astype(dtype=np.uint8)
    image[..., 1] = np.floor(np.arange(0, 128, 128 / (512 * 512))).reshape([512, 512]).astype(dtype=np.uint8)
    image[..., 2] = np.floor(np.arange(0, 64, 64 / (512 * 512))).reshape([512, 512]).astype(dtype=np.uint8)

    decoder = InMemoryCameraFrameDecoder(
        dataset_name="test",
        scene_name="test",
        sensor_name="test_sensor",
        frame_id="1",
        extrinsic=SensorExtrinsic(),
        sensor_pose=SensorPose(),
        annotations={},
        class_maps={},
        intrinsic=SensorIntrinsic(),
        rgba=image,
        image_dimensions=image.shape,
        distortion_lookup=None,
        metadata={},
        date_time=datetime.now(),
    )
    sensor_frame = CameraSensorFrame(decoder=decoder)
    scene_decoder = InMemorySceneDecoder(
        scene_name="test",
    )
    scene = Scene(decoder=scene_decoder)
    image_statistics.update(scene=scene, sensor_frame=sensor_frame)

    hist_red = np.zeros(256, dtype=np.uint32)
    hist_red[...] = 1024
    hist_green = np.zeros(256, dtype=np.uint32)
    hist_green[:128] = 2048
    hist_blue = np.zeros(256, dtype=np.uint32)
    hist_blue[:64] = 4096

    assert np.equal(image_statistics._recorder["histogram_red"], hist_red).all()
    assert np.equal(image_statistics._recorder["histogram_green"], hist_green).all()
    assert np.equal(image_statistics._recorder["histogram_blue"], hist_blue).all()
