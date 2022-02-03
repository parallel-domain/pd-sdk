# from datetime import datetime
# from functools import partial
# from typing import Any, Dict, Generator, Iterable, cast
#
# import pypeln
#
# from paralleldomain import Scene
# from paralleldomain.common.dgp.v1 import annotations_pb2, geometry_pb2, identifiers_pb2, image_pb2, sample_pb2
# from paralleldomain.common.dgp.v1.constants import DirectoryName
# from paralleldomain.common.dgp.v1.utils import datetime_to_timestamp
# from paralleldomain.encoding.dgp.v1.encoder_steps.encoder_step import EncoderStep
# from paralleldomain.encoding.dgp.v1.utils import _attribute_key_dump, _attribute_value_dump
# from paralleldomain.model.annotation import AnnotationTypes, BoundingBox2D
# from paralleldomain.model.image import Image
# from paralleldomain.model.sensor import CameraSensorFrame, FilePathedDataType
# from paralleldomain.utilities import fsio
# from paralleldomain.utilities.any_path import AnyPath
# from paralleldomain.utilities.fsio import relative_path
#
#
# class CameraSensorEncoderStep(EncoderStep):
#     def __init__(
#         self,
#         fs_copy: bool,
#         workers: int = 1,
#         in_queue_size: int = 4,
#     ):
#         self.in_queue_size = in_queue_size
#         self.workers = workers
#         self.fs_copy = fs_copy
#
#     def encode_camera_sensor(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
#         sensor_frame = self._get_camera_frame_from_input_dict(input_dict=input_dict)
#         if sensor_frame is not None:
#             if "pose" not in input_dict:
#                 input_dict["pose"] = dict()
#
#             quaternion = sensor_frame.pose.quaternion
#             input_dict["date_time"] = camera_frame.date_time
#             input_dict["image_height"] = sensor_frame.image.height
#             input_dict["image_width"] = sensor_frame.image.width
#             input_dict["pose"]["translation"] = sensor_frame.pose.translation.tolist()
#             input_dict["pose"]["translation"] = sensor_frame.pose.translation.tolist()
#             input_dict["pose"]["rotation"] = [quaternion.w, quaternion.x, quaternion.y, quaternion.z]
#         return input_dict
#
#     def apply(self, scene: Scene, input_stage: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
#         # assume inputs are yielded sensor by sensor
#         # so if sensor name changes all samples of this sensor have been seen
#         stage = pypeln.thread.ordered(input_stage)
#
#         stage = pypeln.thread.map(
#             f=self.encode_camera_sensor, stage=stage, workers=self.workers, maxsize=self.in_queue_size
#         )
#         return stage
#
#
# class SensorInformationAggregator:
#     def __init__(self):
#         self.scene_data_dtos = list()
#
#     def __call__(self, input_dict: Dict[str, Any]):
#         # rame_id, result_dict = res.result()
#         # camera_frame = camera.get_frame(frame_id)
#         # check if its a camera
#         if "camera_frame_info" in input_dict:
#             sensor_data = input_dict["sensor_data"]
#             annotations = input_dict["annotations"]
#             frame_id = input_dict["camera_frame_info"]["frame_id"]
#             metadata = input_dict["metadata"]
#             date_time = input_dict["date_time"]
#             image_width = input_dict["image_width"]
#             image_height = input_dict["image_height"]
#             scene_output_path = input_dict["scene_output_path"]
#             target_sensor_name = input_dict["target_sensor_name"]
#             translation = input_dict["pose"]["translation"]
#             rotation = input_dict["pose"]["rotation"]
#
#             scene_datum_dto = image_pb2.Image(
#                 filename=relative_path(path=scene_output_path, start=sensor_data[DirectoryName.RGB]).as_posix(),
#                 height=image_height,
#                 width=image_width,
#                 channels=4,
#                 annotations={
#                     int(k): relative_path(path=scene_output_path, start=v).as_posix()
#                     for k, v in annotations.items()
#                     if v is not None
#                 },
#                 pose=geometry_pb2.Pose(
#                     translation=geometry_pb2.Vector3(
#                         x=translation[0],
#                         y=translation[1],
#                         z=translation[2],
#                     ),
#                     rotation=geometry_pb2.Quaternion(
#                         qw=rotation[0],
#                         qx=rotation[1],
#                         qy=rotation[2],
#                         qz=rotation[3],
#                     ),
#                 ),
#                 metadata={str(k): v for k, v in metadata.items()},
#             )
#             # noinspection PyTypeChecker
#             self.scene_data_dtos.append(
#                 sample_pb2.Datum(
#                     id=identifiers_pb2.DatumId(
#                         log="",
#                         name=target_sensor_name,
#                         timestamp=datetime_to_timestamp(dt=date_time),
#                         index=int(frame_id),
#                     ),
#                     key="",
#                     datum=sample_pb2.DatumValue(image=scene_datum_dto),
#                     next_key="",
#                     prev_key="",
#                 )
#             )
#
#     def _process_encode_camera_results(
#         self,
#         camera_name: str,
#         camera_encoding_futures: Set[Future],
#         # camera_encoding_results: Iterator[Tuple[str, Dict[str, Dict[str, Future]]]],
#     ) -> Tuple[str, Dict[str, sample_pb2.Datum]]:
#         scene_data_dtos = []
#
#         camera = self._scene.get_sensor(camera_name)
#         for res in concurrent.futures.as_completed(camera_encoding_futures):
#             frame_id, result_dict = res.result()
#             camera_frame = camera.get_frame(frame_id)
#             sensor_data = result_dict["sensor_data"]
#             annotations = result_dict["annotations"]
#             metadata = result_dict["metadata"]
#
#             scene_datum_dto = image_pb2.Image(
#                 filename=self._relative_path(sensor_data[DirectoryName.RGB].result()).as_posix(),
#                 height=camera_frame.image.height,
#                 width=camera_frame.image.width,
#                 channels=4,
#                 annotations={
#                     int(k): self._relative_path(v.result()).as_posix()
#                     for k, v in annotations.items() if v is not None
#                 },
#                 pose=geometry_pb2.Pose(
#                     translation=geometry_pb2.Vector3(
#                         x=camera_frame.pose.translation[0],
#                         y=camera_frame.pose.translation[1],
#                         z=camera_frame.pose.translation[2],
#                     ),
#                     rotation=geometry_pb2.Quaternion(
#                         qw=camera_frame.pose.quaternion.w,
#                         qx=camera_frame.pose.quaternion.x,
#                         qy=camera_frame.pose.quaternion.y,
#                         qz=camera_frame.pose.quaternion.z,
#                     ),
#                 ),
#                 metadata={str(k): v for k, v in metadata.items()},
#             )
#             # noinspection PyTypeChecker
#             scene_data_dtos.append(
#                 sample_pb2.Datum(
#                     id=identifiers_pb2.DatumId(
#                         log="",
#                         name=camera_frame.sensor_name,
#                         timestamp=datetime_to_timestamp(dt=camera_frame.date_time),
#                         index=int(camera_frame.frame_id),
#                     ),
#                     key="",
#                     datum=sample_pb2.DatumValue(image=scene_datum_dto),
#                     next_key="",
#                     prev_key="",
#                 )
#             )
#
#         scene_data_count = len(scene_data_dtos)
#         # noinspection InsecureHash
#         keys = [hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest() for _ in range(scene_data_count)]
#
#         for idx, scene_data_dto in enumerate(sorted(scene_data_dtos, key=lambda x: x.id.timestamp.ToDatetime())):
#             prev_key = keys[idx - 1] if idx > 0 else ""
#             key = keys[idx]
#             next_key = keys[idx + 1] if idx < (scene_data_count - 1) else ""
#
#             scene_data_dto.prev_key = prev_key
#             scene_data_dto.key = key
#             scene_data_dto.next_key = next_key
#
#         return camera_name, {str(sd.id.index): sd for sd in scene_data_dtos}
