from paralleldomain import Dataset
from paralleldomain.decoding.decoder import DatasetDecoder
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame

TEST_SET_CAM_NAMES = [
    "camera_front",
    "camera_rear",
    "virtual_lidar_rear_camera_0",
    "virtual_lidar_rear_camera_1",
    "virtual_lidar_rear_camera_2",
    "virtual_lidar_front_camera_0",
    "virtual_lidar_front_camera_1",
    "virtual_lidar_front_camera_2",
]
TEST_SET_LIDAR_NAMES = [
    "lidar_front",
    "lidar_rear",
]


def test_can_load_dataset_from_path(decoder: DatasetDecoder):
    dataset = decoder.get_dataset()

    assert len(dataset.scene_names) > 0


def test_can_load_scene(decoder: DatasetDecoder):
    dataset = decoder.get_dataset()

    scene = dataset.get_scene(scene_name=dataset.scene_names[0])
    assert scene is not None
    assert scene.name == dataset.scene_names[0]


class TestDatasetSensors:
    def test_lidar_frame_loading(self, dataset: Dataset):
        num_frames = 0
        for scene in dataset.unordered_scenes.values():
            for sensor in scene.lidars:
                num_frames += len(sensor.frame_ids)

        lidar_frames = list(dataset.lidar_frames)
        assert len(lidar_frames) == num_frames
        assert len(lidar_frames) == dataset.number_of_lidar_frames
        for frame in lidar_frames:
            assert frame is not None
            assert isinstance(frame, LidarSensorFrame)

    def test_camera_frame_loading(self, dataset: Dataset):
        num_frames = 0
        for scene in dataset.unordered_scenes.values():
            for sensor in scene.cameras:
                num_frames += len(sensor.frame_ids)

        cameras_frames = list(dataset.camera_frames)
        assert len(cameras_frames) == num_frames
        assert len(cameras_frames) == dataset.number_of_camera_frames
        for frame in cameras_frames:
            assert frame is not None
            assert isinstance(frame, CameraSensorFrame)

    def test_sensor_frame_loading(self, dataset: Dataset):
        num_frames = 0
        use_sensor_frameids = set()
        use_sensor_names = set()
        for scene in dataset.unordered_scenes.values():
            use_sensor_names.update(scene.sensor_names[::2])
            for sensor_name in scene.sensor_names[::2]:
                sensor = scene.get_sensor(sensor_name=sensor_name)
                for i, fid in enumerate(sensor.frame_ids):
                    if i % 3 == 0:
                        num_frames += 1
                        use_sensor_frameids.add(fid)

        sensor_frames = list(dataset.get_sensor_frames(sensor_names=use_sensor_names, frame_ids=use_sensor_frameids))
        assert len(sensor_frames) == num_frames
        for frame in sensor_frames:
            assert frame is not None
            assert isinstance(frame, CameraSensorFrame) or isinstance(frame, LidarSensorFrame)

    def test_radar_frame_counts_dont_crash(self, dataset: Dataset):
        num_radar_sensors = dataset.number_of_radar_frames
        assert num_radar_sensors >= 0

    def test_lidar_frame_counts_dont_crash(self, dataset: Dataset):
        num_lidar_sensors = dataset.number_of_lidar_frames
        assert num_lidar_sensors >= 0

    def test_camera_frame_counts_dont_crash(self, dataset: Dataset):
        num_camera_sensors = dataset.number_of_camera_frames
        assert num_camera_sensors >= 0

    def test_camera_names(self, dataset: Dataset):
        camera_names = dataset.camera_names
        assert len(camera_names) == 8
        for name, target_name in zip(
            sorted(camera_names),
            sorted(TEST_SET_CAM_NAMES),
        ):
            assert name == target_name

    def test_lidar_names(self, dataset: Dataset):
        lidar_names = dataset.lidar_names
        assert len(lidar_names) == 2
        for name, target_name in zip(
            sorted(lidar_names),
            sorted(TEST_SET_LIDAR_NAMES),
        ):
            assert name == target_name

    def test_radar_names(self, dataset: Dataset):
        camera_names = dataset.radar_names
        assert len(camera_names) == 0

    def test_ordered_pipeline(self, dataset: Dataset):
        ref_names = []
        for sensor_frame, frame, scene in dataset.sensor_frame_pipeline(shuffle=False):
            ref_names.append(f"{sensor_frame.sensor_name}-{frame.frame_id}-{scene.name}")

        assert len(ref_names) == 10 * 10  # 10 frames * 10 sensors

        for scene_name in ["pd-sdk_test_set"]:
            for frame_id in range(10):
                frame_refs = []
                # since the sensor name order is dependent on how the dataset is stored on disk
                # we just check if the frame order is kept
                for sensor_name in TEST_SET_CAM_NAMES + TEST_SET_LIDAR_NAMES:
                    frame_refs.append(f"{sensor_name}-{frame_id}-{scene_name}")
                for i in range(10):
                    actual_ref = ref_names.pop(0)
                    assert actual_ref in frame_refs

    def test_shuffled_pipeline(self, dataset: Dataset):
        ref_names = []
        for sensor_frame, frame, scene in dataset.sensor_frame_pipeline(shuffle=True):
            ref_names.append(f"{sensor_frame.sensor_name}-{frame.frame_id}-{scene.name}")

        assert len(ref_names) == 10 * 10  # 10 frames * 10 sensors

        items_in_order = []
        for scene_name in ["pd-sdk_test_set"]:
            for frame_id in range(10):
                frame_refs = []
                # since the sensor name order is dependent on how the dataset is stored on disk
                # we just check if the frame order is kept
                for sensor_name in TEST_SET_CAM_NAMES + TEST_SET_LIDAR_NAMES:
                    frame_refs.append(f"{sensor_name}-{frame_id}-{scene_name}")
                for i in range(10):
                    actual_ref = ref_names.pop(0)
                    items_in_order.append(actual_ref in frame_refs)
        assert not all(items_in_order)

    def test_fast_shuffled_pipeline(self, dataset: Dataset):
        ref_names = []
        for sensor_frame, frame, scene in dataset.sensor_frame_pipeline(shuffle=True, fast_shuffle=True):
            ref_names.append(f"{sensor_frame.sensor_name}-{frame.frame_id}-{scene.name}")

        assert len(ref_names) == 10 * 10  # 10 frames * 10 sensors

        items_in_order = []
        for scene_name in ["pd-sdk_test_set"]:
            for frame_id in range(10):
                frame_refs = []
                # since the sensor name order is dependent on how the dataset is stored on disk
                # we just check if the frame order is kept
                for sensor_name in TEST_SET_CAM_NAMES + TEST_SET_LIDAR_NAMES:
                    frame_refs.append(f"{sensor_name}-{frame_id}-{scene_name}")
                for i in range(10):
                    actual_ref = ref_names.pop(0)
                    items_in_order.append(actual_ref in frame_refs)
        assert not all(items_in_order)

    def test_shuffled_concurrent_pipeline(self, dataset: Dataset):
        ref_names = []
        for sensor_frame, frame, scene in dataset.sensor_frame_pipeline(shuffle=True, concurrent=True):
            ref_names.append(f"{sensor_frame.sensor_name}-{frame.frame_id}-{scene.name}")

        assert len(ref_names) == 10 * 10  # 10 frames * 10 sensors

        items_in_order = []
        for scene_name in ["pd-sdk_test_set"]:
            for frame_id in range(10):
                frame_refs = []
                # since the sensor name order is dependent on how the dataset is stored on disk
                # we just check if the frame order is kept
                for sensor_name in TEST_SET_CAM_NAMES + TEST_SET_LIDAR_NAMES:
                    frame_refs.append(f"{sensor_name}-{frame_id}-{scene_name}")
                for i in range(10):
                    actual_ref = ref_names.pop(0)
                    items_in_order.append(actual_ref in frame_refs)
        assert not all(items_in_order)
