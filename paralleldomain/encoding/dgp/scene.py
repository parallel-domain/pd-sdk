import hashlib
import logging
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing.pool import ApplyResult, AsyncResult
from typing import Dict, Iterator, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np

from paralleldomain.common.dgp.v0.constants import ANNOTATION_TYPE_MAP_INV
from paralleldomain.common.dgp.v0.dtos import (
    AnnotationsBoundingBox2DDTO,
    AnnotationsBoundingBox3DDTO,
    BoundingBox2DDTO,
    BoundingBox3DDTO,
    CalibrationDTO,
    CalibrationExtrinsicDTO,
    CalibrationIntrinsicDTO,
    OntologyFileDTO,
    PoseDTO,
    RotationDTO,
    SceneDataDatumImage,
    SceneDataDatumPointCloud,
    SceneDataDatumTypeImage,
    SceneDataDatumTypePointCloud,
    SceneDataDTO,
    SceneDataIdDTO,
    SceneDTO,
    SceneMetadataDTO,
    SceneSampleDTO,
    SceneSampleIdDTO,
    TranslationDTO,
)
from paralleldomain.encoding.dgp.transformer import (
    BoundingBox2DTransformer,
    BoundingBox3DTransformer,
    InstanceSegmentation2DTransformer,
    InstanceSegmentation3DTransformer,
    OpticalFlowTransformer,
    SemanticSegmentation2DTransformer,
    SemanticSegmentation3DTransformer,
)
from paralleldomain.encoding.encoder import SceneEncoder
from paralleldomain.model.annotation import Annotation, AnnotationType, AnnotationTypes, BoundingBox2D, BoundingBox3D
from paralleldomain.model.dataset import SceneDataset
from paralleldomain.model.sensor import CameraModel, TemporalSensorFrame
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import write_json

logger = logging.getLogger(__name__)


class DGPSceneEncoder(SceneEncoder):
    _fisheye_camera_model_map: Dict[str, int] = defaultdict(
        lambda: 2,
        {
            CameraModel.OPENCV_PINHOLE: 0,
            CameraModel.OPENCV_FISHEYE: 1,
        },
    )

    def __init__(
        self,
        dataset: SceneDataset,
        set_name: str,
        output_path: AnyPath,
        camera_names: Optional[Union[List[str], None]] = None,
        lidar_names: Optional[Union[List[str], None]] = None,
        annotation_types: Optional[Union[List[AnnotationType], None]] = None,
    ):
        super().__init__(
            dataset=dataset,
            set_name=set_name,
            output_path=output_path,
            camera_names=camera_names,
            lidar_names=lidar_names,
            annotation_types=annotation_types,
        )

        self._scene = self._sensor_frame_set
        self._reference_timestamp: datetime = self._scene.get_frame(self._scene.ordered_frame_ids[0]).date_time
        self._sim_offset: float = 0.01 * 5  # sim timestep * offset count ; unit: seconds

    def _offset_timestamp(self, compare_datetime: datetime) -> float:
        diff = compare_datetime - self._reference_timestamp
        return diff.total_seconds()

    def _encode_rgb(self, sensor_frame: TemporalSensorFrame) -> AsyncResult:
        output_path = (
            self._output_path
            / "rgb"
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time)+self._sim_offset)*100):018d}.png"  # noqa: E501
        )
        return self._run_async(func=fsio.write_png, obj=sensor_frame.image.rgba, path=output_path)

    def _encode_point_cloud(self, sensor_frame: TemporalSensorFrame) -> AsyncResult:
        output_path = (
            self._output_path
            / "point_cloud"
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time)+self._sim_offset)*100):018d}.npz"  # noqa: E501
        )

        pc = sensor_frame.point_cloud
        pc_dtypes = [
            ("X", "<f4"),
            ("Y", "<f4"),
            ("Z", "<f4"),
            ("INTENSITY", "<f4"),
            ("R", "<f4"),
            ("G", "<f4"),
            ("B", "<f4"),
            ("RING_ID", "<u4"),
            ("TIMESTAMP", "<u8"),
        ]

        row_count = pc.length
        pc_data = np.empty(row_count, dtype=pc_dtypes)

        pc_data["X"] = pc.xyz[:, 0]
        pc_data["Y"] = pc.xyz[:, 1]
        pc_data["Z"] = pc.xyz[:, 2]
        pc_data["INTENSITY"] = pc.intensity[:, 0]
        pc_data["R"] = pc.rgb[:, 0]
        pc_data["G"] = pc.rgb[:, 1]
        pc_data["B"] = pc.rgb[:, 2]
        pc_data["RING_ID"] = pc.ring[:, 0]
        pc_data["TIMESTAMP"] = pc.ts[:, 0]

        return self._run_async(func=fsio.write_npz, obj={"data": pc_data}, path=output_path)

    def _encode_depth(self, sensor_frame: TemporalSensorFrame) -> Union[AsyncResult, None]:
        try:
            depth = sensor_frame.get_annotations(AnnotationTypes.Depth)

            output_path = (
                self._output_path
                / "depth"
                / sensor_frame.sensor_name
                / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time)+self._sim_offset)*100):018d}.npz"  # noqa: E501
            )
            return self._run_async(func=fsio.write_npz, obj=dict(data=depth.depth[..., 0]), path=output_path)
        except ValueError:  # Some renderings can exclude LiDAR from having Depth annotations
            return None

    def _encode_bounding_box_2d(self, box: BoundingBox2D) -> BoundingBox2DDTO:
        return BoundingBox2DDTO.from_bounding_box(box=box)

    def _encode_bounding_boxes_2d(self, sensor_frame: TemporalSensorFrame) -> AsyncResult:
        boxes2d = sensor_frame.get_annotations(AnnotationTypes.BoundingBoxes2D)
        box2d_dto = BoundingBox2DTransformer.transform(objects=[self._encode_bounding_box_2d(b) for b in boxes2d.boxes])
        boxes2d_dto = AnnotationsBoundingBox2DDTO(annotations=box2d_dto)

        output_path = (
            self._output_path
            / "bounding_box_2d"
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time)+self._sim_offset)*100):018d}.json"  # noqa: E501
        )
        return self._run_async(func=fsio.write_json, obj=boxes2d_dto.to_dict(), path=output_path, append_sha1=True)

    def _encode_bounding_box_3d(self, box: BoundingBox3D) -> BoundingBox3DDTO:
        return BoundingBox3DDTO.from_bounding_box(box=box)

    def _encode_bounding_boxes_3d(self, sensor_frame: TemporalSensorFrame) -> AsyncResult:
        boxes3d = sensor_frame.get_annotations(AnnotationTypes.BoundingBoxes3D)
        box3d_dto = BoundingBox3DTransformer.transform(objects=[self._encode_bounding_box_3d(b) for b in boxes3d.boxes])
        boxes3d_dto = AnnotationsBoundingBox3DDTO(annotations=box3d_dto)

        output_path = (
            self._output_path
            / "bounding_box_3d"
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time)+self._sim_offset)*100):018d}.json"  # noqa: E501
        )
        return self._run_async(func=fsio.write_json, obj=boxes3d_dto.to_dict(), path=output_path, append_sha1=True)

    def _encode_semantic_segmentation_2d(self, sensor_frame: TemporalSensorFrame) -> AsyncResult:
        semseg2d = sensor_frame.get_annotations(AnnotationTypes.SemanticSegmentation2D)
        mask_out = SemanticSegmentation2DTransformer.transform(mask=semseg2d.class_ids)

        output_path = (
            self._output_path
            / "semantic_segmentation_2d"
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time)+self._sim_offset)*100):018d}.png"  # noqa: E501
        )

        return self._run_async(func=fsio.write_png, obj=mask_out, path=output_path)

    def _encode_instance_segmentation_2d(self, sensor_frame: TemporalSensorFrame) -> AsyncResult:
        instance2d = sensor_frame.get_annotations(AnnotationTypes.InstanceSegmentation2D)
        mask_out = InstanceSegmentation2DTransformer.transform(mask=instance2d.instance_ids)

        output_path = (
            self._output_path
            / "instance_segmentation_2d"
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time)+self._sim_offset)*100):018d}.png"  # noqa: E501
        )

        return self._run_async(func=fsio.write_png, obj=mask_out, path=output_path)

    def _encode_motion_vectors_2d(self, sensor_frame: TemporalSensorFrame) -> AsyncResult:
        optical_flow = sensor_frame.get_annotations(AnnotationTypes.OpticalFlow)
        mask_out = OpticalFlowTransformer.transform(mask=optical_flow.vectors)

        output_path = (
            self._output_path
            / "motion_vectors_2d"
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time)+self._sim_offset)*100):018d}.png"  # noqa: E501
        )

        return self._run_async(func=fsio.write_png, obj=mask_out, path=output_path)

    def _encode_semantic_segmentation_3d(self, sensor_frame: TemporalSensorFrame) -> AsyncResult:
        semseg3d = sensor_frame.get_annotations(AnnotationTypes.SemanticSegmentation3D)
        mask_out = SemanticSegmentation3DTransformer.transform(mask=semseg3d.class_ids)

        output_path = (
            self._output_path
            / "semantic_segmentation_3d"
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time)+self._sim_offset)*100):018d}.npz"  # noqa: E501
        )

        return self._run_async(func=fsio.write_npz, obj=dict(segmentation=mask_out), path=output_path)

    def _encode_instance_segmentation_3d(self, sensor_frame: TemporalSensorFrame) -> AsyncResult:
        instance3d = sensor_frame.get_annotations(AnnotationTypes.InstanceSegmentation3D)
        mask_out = InstanceSegmentation3DTransformer.transform(mask=instance3d.instance_ids)

        output_path = (
            self._output_path
            / "instance_segmentation_3d"
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time)+self._sim_offset)*100):018d}.npz"  # noqa: E501
        )

        return self._run_async(func=fsio.write_npz, obj=dict(instance=mask_out), path=output_path)

    def _process_encode_camera_results(
        self,
        camera_name: str,
        camera_encoding_results: Iterator[Tuple[str, Dict[str, Dict[str, ApplyResult]]]],
    ) -> Dict[str, SceneDataDTO]:
        scene_data_dtos = []

        camera = self._scene.get_sensor(camera_name)
        for frame_id, result_dict in camera_encoding_results:
            camera_frame = camera.get_frame(frame_id)
            sensor_data = result_dict["sensor_data"]
            annotations = result_dict["annotations"]

            scene_datum_dto = SceneDataDatumTypeImage(
                filename=self._relative_path(sensor_data["rgb"].get()).as_posix(),
                height=camera_frame.image.height,
                width=camera_frame.image.width,
                # replace with Decoder attribute - if accessing .image.rgbs.shape[2] image needs to be loaded :(
                channels=4,
                annotations={
                    k: self._relative_path(v.get()).as_posix() for k, v in annotations.items() if v is not None
                },
                pose=PoseDTO(
                    translation=TranslationDTO(
                        x=camera_frame.pose.translation[0],
                        y=camera_frame.pose.translation[1],
                        z=camera_frame.pose.translation[2],
                    ),
                    rotation=RotationDTO(
                        qw=camera_frame.pose.quaternion.w,
                        qx=camera_frame.pose.quaternion.x,
                        qy=camera_frame.pose.quaternion.y,
                        qz=camera_frame.pose.quaternion.z,
                    ),
                ),
                metadata={},
            )
            # noinspection PyTypeChecker
            scene_data_dtos.append(
                SceneDataDTO(
                    id=SceneDataIdDTO(
                        log="",
                        name=camera_frame.sensor_name,
                        timestamp=camera_frame.date_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        index=str(camera_frame.frame_id),
                    ),
                    key="",
                    datum=SceneDataDatumImage(image=scene_datum_dto),
                    next_key="",
                    prev_key="",
                )
            )

        scene_data_count = len(scene_data_dtos)
        # noinspection InsecureHash
        keys = [hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest() for _ in range(scene_data_count)]

        for idx, scene_data_dto in enumerate(sorted(scene_data_dtos, key=lambda x: x.id.timestamp)):
            prev_key = keys[idx - 1] if idx > 0 else ""
            key = keys[idx]
            next_key = keys[idx + 1] if idx < (scene_data_count - 1) else ""

            scene_data_dto.prev_key = prev_key
            scene_data_dto.key = key
            scene_data_dto.next_key = next_key

        return {sd.id.index: sd for sd in scene_data_dtos}

    def _process_encode_lidar_results(
        self,
        lidar_name: str,
        lidar_encoding_results: Iterator[Tuple[str, Dict[str, Dict[str, ApplyResult]]]],
    ) -> Dict[str, SceneDataDTO]:
        scene_data_dtos = []

        lidar = self._scene.get_sensor(lidar_name)
        for frame_id, result_dict in lidar_encoding_results:
            lidar_frame = lidar.get_frame(frame_id)
            sensor_data = result_dict["sensor_data"]
            annotations = result_dict["annotations"]

            scene_datum_dto = SceneDataDatumTypePointCloud(
                filename=self._relative_path(sensor_data["point_cloud"].get()).as_posix(),
                point_format=["X", "Y", "Z", "INTENSITY", "R", "G", "B", "RING", "TIMESTAMP"],
                annotations={
                    k: self._relative_path(v.get()).as_posix() for k, v in annotations.items() if v is not None
                },
                pose=PoseDTO(
                    translation=TranslationDTO(
                        x=lidar_frame.pose.translation[0],
                        y=lidar_frame.pose.translation[1],
                        z=lidar_frame.pose.translation[2],
                    ),
                    rotation=RotationDTO(
                        qw=lidar_frame.pose.quaternion.w,
                        qx=lidar_frame.pose.quaternion.x,
                        qy=lidar_frame.pose.quaternion.y,
                        qz=lidar_frame.pose.quaternion.z,
                    ),
                ),
                point_fields=[],
                metadata={},
            )
            # noinspection PyTypeChecker
            scene_data_dtos.append(
                SceneDataDTO(
                    id=SceneDataIdDTO(
                        log="",
                        name=lidar_frame.sensor_name,
                        timestamp=lidar_frame.date_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        index=str(lidar_frame.frame_id),
                    ),
                    key="",
                    datum=SceneDataDatumPointCloud(point_cloud=scene_datum_dto),
                    next_key="",
                    prev_key="",
                )
            )

        scene_data_count = len(scene_data_dtos)
        # noinspection InsecureHash
        keys = [hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest() for _ in range(scene_data_count)]

        for idx, scene_data_dto in enumerate(sorted(scene_data_dtos, key=lambda x: x.id.timestamp)):
            prev_key = keys[idx - 1] if idx > 0 else ""
            key = keys[idx]
            next_key = keys[idx + 1] if idx < (scene_data_count - 1) else ""

            scene_data_dto.prev_key = prev_key
            scene_data_dto.key = key
            scene_data_dto.next_key = next_key

        return {sd.id.index: sd for sd in scene_data_dtos}

    def _encode_camera_frame(
        self, camera_frame: TemporalSensorFrame, last_frame: Optional[bool] = False
    ) -> Dict[str, Dict[str, AsyncResult]]:
        return dict(
            annotations={
                "0": self._encode_bounding_boxes_2d(sensor_frame=camera_frame)
                if AnnotationTypes.BoundingBoxes2D in self._annotation_types
                else None,
                "1": self._encode_bounding_boxes_3d(sensor_frame=camera_frame)
                if AnnotationTypes.BoundingBoxes3D in self._annotation_types
                else None,
                "2": self._encode_semantic_segmentation_2d(sensor_frame=camera_frame)
                if AnnotationTypes.SemanticSegmentation2D in self._annotation_types
                else None,
                "4": self._encode_instance_segmentation_2d(sensor_frame=camera_frame)
                if AnnotationTypes.InstanceSegmentation2D in self._annotation_types
                else None,
                "6": self._encode_depth(sensor_frame=camera_frame)
                if AnnotationTypes.Depth in self._annotation_types
                else None,
                "8": self._encode_motion_vectors_2d(sensor_frame=camera_frame)
                if AnnotationTypes.OpticalFlow in self._annotation_types and not last_frame
                else None,
                "10": None,  # surface_normals_2d
            },
            sensor_data={
                "rgb": self._encode_rgb(sensor_frame=camera_frame),
            },
        )

    def _encode_lidar_frame(self, lidar_frame: TemporalSensorFrame) -> Dict[str, Dict[str, AsyncResult]]:
        return dict(
            annotations={
                "1": self._encode_bounding_boxes_3d(sensor_frame=lidar_frame)
                if AnnotationTypes.BoundingBoxes3D in self._annotation_types
                else None,
                "3": self._encode_semantic_segmentation_3d(sensor_frame=lidar_frame)
                if AnnotationTypes.SemanticSegmentation3D in self._annotation_types
                else None,
                "5": self._encode_instance_segmentation_3d(sensor_frame=lidar_frame)
                if AnnotationTypes.InstanceSegmentation3D in self._annotation_types
                else None,
                "6": self._encode_depth(sensor_frame=lidar_frame)
                if AnnotationTypes.Depth in self._annotation_types
                else None,
                "7": None,  # surface_normals_3d
                "9": None,  # motion_vectors_3d
            },
            sensor_data={
                "point_cloud": self._encode_point_cloud(sensor_frame=lidar_frame),
            },
        )

    def _encode_camera(self, camera_name: str) -> Dict[str, SceneDataDTO]:
        with ThreadPoolExecutor(max_workers=10) as camera_frame_executor:
            camera_encoding_results = zip(
                self._scene.ordered_frame_ids,
                camera_frame_executor.map(
                    lambda fid: self._encode_camera_frame(
                        self._scene.get_frame(fid).get_sensor(camera_name),
                        last_frame=(
                            self._scene.ordered_frame_ids.index(fid) == (len(self._scene.ordered_frame_ids) - 1)
                        ),
                    ),
                    self._scene.ordered_frame_ids,
                ),
            )
        return self._process_encode_camera_results(
            camera_name=camera_name, camera_encoding_results=camera_encoding_results
        )

    def _encode_lidar(self, lidar_name: str) -> Dict[str, SceneDataDTO]:
        with ThreadPoolExecutor(max_workers=10) as lidar_frame_executor:
            lidar_encoding_results = zip(
                self._scene.ordered_frame_ids,
                lidar_frame_executor.map(
                    lambda fid: self._encode_lidar_frame(self._scene.get_frame(fid).get_sensor(lidar_name)),
                    self._scene.ordered_frame_ids,
                ),
            )
        return self._process_encode_lidar_results(lidar_name=lidar_name, lidar_encoding_results=lidar_encoding_results)

    def _encode_cameras(self) -> Iterator[Tuple[str, Dict[str, SceneDataDTO]]]:
        with ThreadPoolExecutor(max_workers=4) as camera_executor:
            return zip(self._camera_names, camera_executor.map(self._encode_camera, self._camera_names))

    def _encode_lidars(self) -> Iterator[Tuple[str, Dict[str, SceneDataDTO]]]:
        with ThreadPoolExecutor(max_workers=4) as lidar_executor:
            return zip(self._lidar_names, lidar_executor.map(self._encode_lidar, self._lidar_names))

    def _encode_ontologies(self) -> Dict[str, AsyncResult]:
        ontology_dtos = {
            ANNOTATION_TYPE_MAP_INV[a_type]: OntologyFileDTO.from_class_map(class_map=self._scene.get_class_map(a_type))
            for a_type in self._annotation_types
            if a_type is not Annotation  # equiv: not implemented, yet!
        }

        output_path = self._output_path / "ontology" / ".json"  # noqa: E501

        return {
            k: self._run_async(func=write_json, obj=v.to_dict(), path=output_path, append_sha1=True)
            for k, v in ontology_dtos.items()
        }

    def _encode_calibrations(self) -> AsyncResult:
        sensor_frames = []
        frame_ids = self._scene.ordered_frame_ids
        for sn in self._sensor_names:
            sensor_frames.append(self._scene.get_sensor(sn).get_frame(frame_ids[0]))

        calib_dto = CalibrationDTO(names=[], extrinsics=[], intrinsics=[])

        def get_calibration(sf: TemporalSensorFrame) -> Tuple[str, CalibrationExtrinsicDTO, CalibrationIntrinsicDTO]:
            intr = sf.intrinsic
            extr = sf.extrinsic

            calib_dto_extrinsic = CalibrationExtrinsicDTO(
                translation=TranslationDTO(x=extr.translation[0], y=extr.translation[1], z=extr.translation[2]),
                rotation=RotationDTO(
                    qw=extr.quaternion.w, qx=extr.quaternion.x, qy=extr.quaternion.y, qz=extr.quaternion.z
                ),
            )

            calib_dto_intrinsic = CalibrationIntrinsicDTO(
                fx=intr.fx,
                fy=intr.fy,
                cx=intr.cx,
                cy=intr.cy,
                skew=intr.skew,
                fov=intr.fov,
                k1=intr.k1,
                k2=intr.k2,
                k3=intr.k3,
                k4=intr.k4,
                k5=intr.k5,
                k6=intr.k6,
                p1=intr.p1,
                p2=intr.p2,
                fisheye=self._fisheye_camera_model_map[intr.camera_model],
            )

            return (sf.sensor_name, calib_dto_extrinsic, calib_dto_intrinsic)

        res = map(get_calibration, sensor_frames)

        for r_name, r_extrinsic, r_intrinsic in res:
            calib_dto.names.append(r_name)
            calib_dto.extrinsics.append(r_extrinsic)
            calib_dto.intrinsics.append(r_intrinsic)

        output_path = self._output_path / "calibration" / ".json"  # noqa: E501
        return self._run_async(func=fsio.write_json, obj=calib_dto.to_dict(), path=output_path, append_sha1=True)

    def _encode_scene_json(
        self,
        scene_sensor_data: Dict[str, Dict[str, SceneDataDTO]],
        calibration_file: AsyncResult,
        ontologies_files: Dict[str, AsyncResult],
    ) -> AnyPath:
        scene_data = []
        scene_samples = []
        for fid in self._scene.ordered_frame_ids:
            frame = self._scene.get_frame(fid)
            frame_data = [
                scene_sensor_data[sn][fid] for sn in sorted(scene_sensor_data.keys()) if fid in scene_sensor_data[sn]
            ]
            scene_data.extend(frame_data)
            scene_samples.append(
                SceneSampleDTO(
                    id=SceneSampleIdDTO(
                        log="",
                        timestamp=frame.date_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        name="",
                        index=frame.frame_id,
                    ),
                    datum_keys=[d.key for d in frame_data],
                    calibration_key=calibration_file.get().stem,
                    metadata={},
                )
            )

        scene_dto = SceneDTO(
            name=self._scene.name,
            description=self._scene.description,
            log="",
            ontologies={k: v.get().stem for k, v in ontologies_files.items()},
            metadata=SceneMetadataDTO.from_dict(self._scene.metadata),
            samples=scene_samples,
            data=scene_data,
        )

        output_path = self._output_path / "scene.json"  # noqa: E501
        return fsio.write_json(obj=scene_dto.to_dict(), path=output_path, append_sha1=True)

    def _encode_sensors(self) -> Dict[str, Dict[str, SceneDataDTO]]:
        scene_camera_data = self._encode_cameras()
        scene_lidar_data = self._encode_lidars()

        scene_sensor_data = {}
        scene_sensor_data.update(dict(scene_camera_data))
        scene_sensor_data.update(dict(scene_lidar_data))

        return scene_sensor_data

    def _run_encoding(self) -> AnyPath:
        scene_sensor_data = self._encode_sensors()
        calibration_file = self._encode_calibrations()
        ontologies_files = self._encode_ontologies()
        return self._encode_scene_json(
            scene_sensor_data=scene_sensor_data, calibration_file=calibration_file, ontologies_files=ontologies_files
        )

    def _prepare_output_directories(self) -> None:
        super()._prepare_output_directories()
        if not urlparse(str(self._output_path)).scheme:  # Local FS - needs existing directories
            (self._output_path / "calibration").mkdir(exist_ok=True, parents=True)
            (self._output_path / "ontology").mkdir(exist_ok=True, parents=True)
            for camera_name in self._camera_names:
                (self._output_path / "rgb" / camera_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.BoundingBoxes2D in self._annotation_types:
                    (self._output_path / "bounding_box_2d" / camera_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.BoundingBoxes3D in self._annotation_types:
                    (self._output_path / "bounding_box_3d" / camera_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.SemanticSegmentation2D in self._annotation_types:
                    (self._output_path / "semantic_segmentation_2d" / camera_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.InstanceSegmentation2D in self._annotation_types:
                    (self._output_path / "instance_segmentation_2d" / camera_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.OpticalFlow in self._annotation_types:
                    (self._output_path / "motion_vectors_2d" / camera_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.Depth in self._annotation_types:
                    (self._output_path / "depth" / camera_name).mkdir(exist_ok=True, parents=True)
            for lidar_name in self._lidar_names:
                (self._output_path / "point_cloud" / lidar_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.BoundingBoxes3D in self._annotation_types:
                    (self._output_path / "bounding_box_3d" / lidar_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.Depth in self._annotation_types:
                    (self._output_path / "depth" / lidar_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.SemanticSegmentation3D in self._annotation_types:
                    (self._output_path / "semantic_segmentation_3d" / lidar_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.InstanceSegmentation3D in self._annotation_types:
                    (self._output_path / "instance_segmentation_3d" / lidar_name).mkdir(exist_ok=True, parents=True)
