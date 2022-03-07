from paralleldomain import Dataset
from paralleldomain.decoding.decoder import DatasetDecoder
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame


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
