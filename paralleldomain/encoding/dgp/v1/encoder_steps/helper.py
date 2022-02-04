import abc
from datetime import datetime
from typing import Any, Dict, Generator, Iterable, Optional, cast

from paralleldomain import Scene
from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, SensorFrame
from paralleldomain.utilities.any_path import AnyPath


class EncoderStepHelper:
    @staticmethod
    def _offset_timestamp(compare_datetime: datetime, reference_timestamp: datetime) -> float:
        diff = compare_datetime - reference_timestamp
        return diff.total_seconds()

    @staticmethod
    def _get_offset_timestamp_file_name(sensor_frame: SensorFrame[datetime], input_dict: Dict[str, Any]) -> str:
        scene_reference_timestamp = input_dict["scene_reference_timestamp"]
        sim_offset = input_dict["sim_offset"]
        offset = EncoderStepHelper._offset_timestamp(
            compare_datetime=sensor_frame.date_time, reference_timestamp=scene_reference_timestamp
        )
        return f"{round((offset + sim_offset) * 100):018d}"

    @staticmethod
    def _get_dgpv1_file_output_path(
        sensor_frame: SensorFrame[datetime], input_dict: Dict[str, Any], file_suffix: str, directory_name: str
    ) -> AnyPath:
        scene_output_path = input_dict["scene_output_path"]
        target_sensor_name = input_dict["target_sensor_name"]
        file_name = EncoderStepHelper._get_offset_timestamp_file_name(sensor_frame=sensor_frame, input_dict=input_dict)
        if file_suffix.startswith("."):
            file_suffix = file_suffix[1:]
        output_path = (
            scene_output_path
            / directory_name
            / target_sensor_name
            / f"{file_name}.{file_suffix}"
            # noqa: E501
        )
        return output_path

    def _scene_from_input_dict(self, input_dict: Dict[str, Any]) -> Scene:
        info_dict = input_dict.get("camera_frame_info", input_dict.get("lidar_frame_info"))
        return self._scene_from_info_dict(info_dict=info_dict)

    def _scene_from_info_dict(self, info_dict: Dict[str, Any]) -> Scene:
        scene_name = info_dict["scene_name"]
        # dataset_path = info_dict["dataset_path"]
        dataset_format = info_dict["dataset_format"]
        decoder_kwargs = info_dict["decoder_kwargs"]
        dataset = decode_dataset(dataset_format=dataset_format, **decoder_kwargs)
        return dataset.get_scene(scene_name=scene_name)

    def _get_camera_frame_from_input_dict(self, input_dict: Dict[str, Any]) -> Optional[CameraSensorFrame[datetime]]:
        if "camera_frame" in input_dict:
            sensor_frame = input_dict["camera_frame"]
            if isinstance(sensor_frame, CameraSensorFrame):
                return sensor_frame
        elif "camera_frame_info" in input_dict:
            scene = self._scene_from_info_dict(info_dict=input_dict["camera_frame_info"])
            sensor_name = input_dict["camera_frame_info"]["sensor_name"]
            frame_id = input_dict["camera_frame_info"]["frame_id"]
            return scene.get_camera_sensor(camera_name=sensor_name).get_frame(frame_id=frame_id)
        return None

    def _get_lidar_frame_from_input_dict(self, input_dict: Dict[str, Any]) -> Optional[LidarSensorFrame[datetime]]:
        if "lidar_frame" in input_dict:
            sensor_frame = input_dict["lidar_frame"]
            if isinstance(sensor_frame, LidarSensorFrame):
                return sensor_frame
        elif "lidar_frame_info" in input_dict:
            scene = self._scene_from_info_dict(info_dict=input_dict["lidar_frame_info"])
            sensor_name = input_dict["lidar_frame_info"]["sensor_name"]
            frame_id = input_dict["lidar_frame_info"]["frame_id"]
            return scene.get_lidar_sensor(lidar_name=sensor_name).get_frame(frame_id=frame_id)
        return None
