from typing import Dict

import pytest

from paralleldomain import Scene
from paralleldomain.model.ego import EgoFrame, EgoPose
from paralleldomain.model.frame import Frame


@pytest.fixture()
def frame(scene: Scene) -> Frame:
    frame_id = scene.frame_ids[0]
    return scene.get_frame(frame_id=frame_id)


class TestSceneFrames:
    def test_frame_camera_names_are_loadable(self, frame: Frame):
        camera_names = frame.camera_names
        assert len(camera_names) > 0

    def test_frame_lidar_names_are_loadable(self, frame: Frame):
        lidar_names = frame.lidar_names
        assert len(lidar_names) > 0

    def test_frame_sensors_names_are_loadable(self, frame: Frame):
        sensor_names = frame.sensor_names
        assert len(sensor_names) > 0

    def test_frame_sensors_are_loadable(self, frame: Frame):
        sensor_names = frame.sensor_names
        assert len(sensor_names) > 0
        sensor_frames = list(frame.sensor_frames)
        assert len(sensor_frames) == len(sensor_names)

    def test_frame_lidars_are_loadable(self, frame: Frame):
        lidar_names = frame.lidar_names
        assert len(lidar_names) > 0
        lidar_frames = list(frame.lidar_frames)
        assert len(lidar_frames) == len(lidar_names)

    def test_frame_cameras_are_loadable(self, frame: Frame):
        camera_names = frame.camera_names
        assert len(camera_names) > 0
        camera_frames = list(frame.camera_frames)
        assert len(camera_frames) == len(camera_names)

    def test_frame_ego_frame_is_loadable(self, frame: Frame):
        ego_frame = frame.ego_frame
        assert ego_frame is not None
        assert isinstance(ego_frame, EgoFrame)
        pose = ego_frame.pose
        assert isinstance(pose, EgoPose)

    def test_frame_metadata_is_loadable(self, frame: Frame):
        metadata = frame.metadata
        assert metadata is not None
        assert isinstance(metadata, Dict)

    """
    You can use this test method to plot the locations of the vehicle and the orientation
    @pytest.skip
    def test_print_trajectory(self, scene: Scene):
        import numpy as np
        import cv2
        blank_image = np.zeros((1000, 1000, 3), np.uint8) * 255
        starts = list()
        ends = list()
        min_x = 100000
        max_x = 0
        min_y = 100000
        max_y = 0

        for i, frame_id in enumerate(scene.frame_ids[::4]):
            ego_pose = scene.get_frame(frame_id).ego_frame.pose
            front = np.array([0.5, 0., 0., 1.])
            front = ego_pose @ front
            # front = (17 * front) + 600
            # pos = (17 * ego_pose.translation) + 600
            pos = ego_pose.translation
            start_pos = (pos[0], pos[1])
            end_pos = (front[0], front[1])
            min_x = min(min_x, start_pos[0], end_pos[0])
            min_y = min(min_y, start_pos[1], end_pos[1])
            max_x = max(max_x, start_pos[0], end_pos[0])
            max_y = max(max_y, start_pos[1], end_pos[1])

            starts.append(start_pos)
            ends.append(end_pos)

        scale_x = 800 / (max_x - min_x)
        scale_y = 800 / (max_y - min_y)
        for i, (start, end) in enumerate(zip(starts, ends)):
            start = (int((start[0] - min_x) * scale_x + 100), int((start[1] - min_y) * scale_y + 100))
            end = (int((end[0] - min_x) * scale_x + 100), int((end[1] - min_y) * scale_y + 100))
            blank_image = cv2.line(blank_image, start, end, (i * 5, 0, max(0, 255 - 3 * i)), 1)
            cv2.circle(blank_image, start, 1, (i * 5, 0, max(0, 255 - 3 * i)), -1)

        cv2.imshow("window_name", blank_image)
        cv2.waitKey()
    """
