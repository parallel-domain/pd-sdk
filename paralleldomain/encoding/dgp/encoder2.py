import argparse
import logging
from urllib.parse import urlparse

import numpy as np

from paralleldomain.encoding.dgp.dtos import (
    AnnotationsBoundingBox2DDTO,
    AnnotationsBoundingBox3DDTO,
    BoundingBox2DDTO,
    BoundingBox3DDTO,
)
from paralleldomain.encoding.encoder import DatasetEncoder, MaskFilter, ObjectFilter, SceneEncoder
from paralleldomain.encoding.utils import fsio
from paralleldomain.encoding.utils.log import setup_loggers
from paralleldomain.encoding.utils.mask import encode_2int16_as_rgba8, encode_int32_as_rgb8
from paralleldomain.model.annotation import AnnotationTypes, BoundingBox2D, BoundingBox3D
from paralleldomain.model.sensor import SensorFrame

logger = logging.getLogger(__name__)


class BoundingBox2DFilter(ObjectFilter):
    ...


class BoundingBox3DFilter(ObjectFilter):
    ...


class SemanticSegmentation2DFilter(MaskFilter):
    @staticmethod
    def transform(mask: np.ndarray) -> np.ndarray:
        return encode_int32_as_rgb8(mask)


class InstanceSegmentation2DFilter(SemanticSegmentation2DFilter):
    ...


class OpticalFlowFilter(MaskFilter):
    @staticmethod
    def transform(mask: np.ndarray) -> np.ndarray:
        return encode_2int16_as_rgba8(mask)


class SemanticSegmentation3DFilter(MaskFilter):
    @staticmethod
    def transform(mask: np.ndarray) -> np.ndarray:
        return mask.astype(np.uint32)


class InstanceSegmentation3DFilter(SemanticSegmentation3DFilter):
    ...


class DGPSceneEncoder(SceneEncoder):
    def _encode_rgb(self, sensor_frame: SensorFrame):
        output_path = self._output_path / "rgb" / sensor_frame.sensor_name / f"{int(sensor_frame.frame_id):018d}.png"
        return self._run_async(func=fsio.write_png, obj=sensor_frame.image.rgba, path=output_path)

    def _encode_point_cloud(self, sensor_frame: SensorFrame):
        output_path = (
            self._output_path / "point_cloud" / sensor_frame.sensor_name / f"{int(sensor_frame.frame_id):018d}.npz"
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

    def _encode_depth(self, sensor_frame: SensorFrame):
        depth = sensor_frame.get_annotations(AnnotationTypes.Depth)

        output_path = self._output_path / "depth" / sensor_frame.sensor_name / f"{int(sensor_frame.frame_id):018d}.npz"
        return self._run_async(func=fsio.write_npz, obj=dict(data=depth.depth[..., 0]), path=output_path)

    def _encode_bounding_box_2d(self, box: BoundingBox2D) -> BoundingBox2DDTO:
        return BoundingBox2DDTO.from_BoundingBox2D(box)

    def _encode_bounding_boxes_2d(self, sensor_frame: SensorFrame):
        boxes2d = sensor_frame.get_annotations(AnnotationTypes.BoundingBoxes2D)
        box2d_dto = BoundingBox2DFilter.run([self._encode_bounding_box_2d(b) for b in boxes2d.boxes])
        boxes2d_dto = AnnotationsBoundingBox2DDTO(annotations=box2d_dto)

        output_path = (
            self._output_path / "bounding_box_2d" / sensor_frame.sensor_name / f"{int(sensor_frame.frame_id):018d}.json"
        )
        return self._run_async(func=fsio.write_json, obj=boxes2d_dto.to_dict(), path=output_path, append_sha1=True)

    def _encode_bounding_box_3d(self, box: BoundingBox3D) -> BoundingBox3DDTO:
        return BoundingBox3DDTO.from_BoundingBox3D(box)

    def _encode_bounding_boxes_3d(self, sensor_frame: SensorFrame):
        boxes3d = sensor_frame.get_annotations(AnnotationTypes.BoundingBoxes3D)
        box3d_dto = BoundingBox3DFilter.run(objects=[self._encode_bounding_box_3d(b) for b in boxes3d.boxes])
        boxes3d_dto = AnnotationsBoundingBox3DDTO(annotations=box3d_dto)

        output_path = (
            self._output_path / "bounding_box_3d" / sensor_frame.sensor_name / f"{int(sensor_frame.frame_id):018d}.json"
        )
        return self._run_async(func=fsio.write_json, obj=boxes3d_dto.to_dict(), path=output_path)

    def _encode_semantic_segmentation_2d(self, sensor_frame: SensorFrame):
        semseg2d = sensor_frame.get_annotations(AnnotationTypes.SemanticSegmentation2D)
        mask_out = SemanticSegmentation2DFilter.run(mask=semseg2d.class_ids)

        output_path = (
            self._output_path
            / "semantic_segmentation_2d"
            / sensor_frame.sensor_name
            / f"{int(sensor_frame.frame_id):018d}.png"
        )

        return self._run_async(func=fsio.write_png, obj=mask_out, path=output_path)

    def _encode_instance_segmentation_2d(self, sensor_frame: SensorFrame):
        instance2d = sensor_frame.get_annotations(AnnotationTypes.InstanceSegmentation2D)
        mask_out = InstanceSegmentation2DFilter.run(mask=instance2d.instance_ids)

        output_path = (
            self._output_path
            / "instance_segmentation_2d"
            / sensor_frame.sensor_name
            / f"{int(sensor_frame.frame_id):018d}.png"
        )

        return self._run_async(func=fsio.write_png, obj=mask_out, path=output_path)

    def _encode_motion_vectors_2d(self, sensor_frame: SensorFrame):
        optical_flow = sensor_frame.get_annotations(AnnotationTypes.OpticalFlow)
        mask_out = OpticalFlowFilter.run(mask=optical_flow.vectors)

        output_path = (
            self._output_path
            / "motion_vectors_2d"
            / sensor_frame.sensor_name
            / f"{int(sensor_frame.frame_id):018d}.png"
        )

        return self._run_async(func=fsio.write_png, obj=mask_out, path=output_path)

    def _encode_semantic_segmentation_3d(self, sensor_frame: SensorFrame):
        semseg3d = sensor_frame.get_annotations(AnnotationTypes.SemanticSegmentation3D)
        mask_out = SemanticSegmentation3DFilter.run(semseg3d.class_ids)

        output_path = (
            self._output_path
            / "semantic_segmentation_3d"
            / sensor_frame.sensor_name
            / f"{int(sensor_frame.frame_id):018d}.npz"
        )

        return self._run_async(func=fsio.write_npz, obj=dict(segmentation=mask_out), path=output_path)

    def _encode_instance_segmentation_3d(self, sensor_frame: SensorFrame):
        instance3d = sensor_frame.get_annotations(AnnotationTypes.InstanceSegmentation3D)
        mask_out = InstanceSegmentation3DFilter.run(instance3d.instance_ids)

        output_path = (
            self._output_path
            / "instance_segmentation_3d"
            / sensor_frame.sensor_name
            / f"{int(sensor_frame.frame_id):018d}.npz"
        )

        return self._run_async(func=fsio.write_npz, obj=dict(instance=mask_out), path=output_path)

    def _encode_camera_frame(self, camera_frame: SensorFrame):
        path_bounding_boxes_2d = self._encode_bounding_boxes_2d(sensor_frame=camera_frame)  # noqa: F841
        path_depth = self._encode_depth(sensor_frame=camera_frame)  # noqa: F841
        path_motion_vectors_2d = self._encode_motion_vectors_2d(sensor_frame=camera_frame)  # noqa: F841
        path_semantic_segmentation_2d = self._encode_semantic_segmentation_2d(sensor_frame=camera_frame)  # noqa: F841
        path_instance_segmentation_2d = self._encode_instance_segmentation_2d(sensor_frame=camera_frame)  # noqa: F841
        path_bounding_boxes_3d = self._encode_bounding_boxes_3d(sensor_frame=camera_frame)  # noqa: F841
        path_rgb = self._encode_rgb(sensor_frame=camera_frame)  # noqa: F841

    def _encode_lidar_frame(self, lidar_frame: SensorFrame):
        path_instance_segmentation_3d = self._encode_instance_segmentation_3d(sensor_frame=lidar_frame)  # noqa: F841
        path_semantic_segmentation_3d = self._encode_semantic_segmentation_3d(sensor_frame=lidar_frame)  # noqa: F841
        path_depth = self._encode_depth(sensor_frame=lidar_frame)  # noqa: F841
        path_bounding_boxes_3d = self._encode_bounding_boxes_3d(sensor_frame=lidar_frame)  # noqa: F841
        path_point_cloud = self._encode_point_cloud(sensor_frame=lidar_frame)  # noqa: F841

    def _prepare_output_directories(self) -> None:
        super()._prepare_output_directories()
        if not urlparse(str(self._output_path)).scheme:  # Local FS - needs existing directories
            for camera_name in self._scene.camera_names:
                (self._output_path / "rgb" / camera_name).mkdir(exist_ok=True, parents=True)
                (self._output_path / "bounding_box_2d" / camera_name).mkdir(exist_ok=True, parents=True)
                (self._output_path / "bounding_box_3d" / camera_name).mkdir(exist_ok=True, parents=True)
                (self._output_path / "semantic_segmentation_2d" / camera_name).mkdir(exist_ok=True, parents=True)
                (self._output_path / "instance_segmentation_2d" / camera_name).mkdir(exist_ok=True, parents=True)
                (self._output_path / "motion_vectors_2d" / camera_name).mkdir(exist_ok=True, parents=True)
                (self._output_path / "depth" / camera_name).mkdir(exist_ok=True, parents=True)
            for lidar_name in self._scene.lidar_names:
                (self._output_path / "point_cloud" / lidar_name).mkdir(exist_ok=True, parents=True)
                (self._output_path / "bounding_box_3d" / lidar_name).mkdir(exist_ok=True, parents=True)
                (self._output_path / "depth" / lidar_name).mkdir(exist_ok=True, parents=True)
                (self._output_path / "semantic_segmentation_3d" / lidar_name).mkdir(exist_ok=True, parents=True)
                (self._output_path / "instance_segmentation_3d" / lidar_name).mkdir(exist_ok=True, parents=True)


class DGPDatasetEncoder(DatasetEncoder):
    scene_encoder = DGPSceneEncoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a data encoders")
    parser.add_argument("-i", "--input", type=str, help="A local or cloud path to a DGP dataset", required=True)
    parser.add_argument("-o", "--output", type=str, help="A local or cloud path for the encoded dataset", required=True)
    parser.add_argument(
        "--scene_names",
        nargs="*",
        type=str,
        help="""Define one or multiple specific scenes to be processed.
                When provided, overwrites any scene_start and scene_stop arguments""",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--scene_start",
        type=int,
        help="An integer defining the start index for scene processing",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--scene_stop",
        type=int,
        help="An integer defining the stop index for scene processing",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--n_parallel",
        type=int,
        help="Define how many scenes should be processed in parallel",
        required=False,
        default=1,
    )

    args = parser.parse_args()

    setup_loggers([__name__, DGPDatasetEncoder.__name__, DGPSceneEncoder.__name__, "fsio"], log_level=logging.DEBUG)

    DGPDatasetEncoder(
        input_path=args.input,
        output_path=args.output,
        scene_names=args.scene_names,
        scene_start=args.scene_start,
        scene_stop=args.scene_stop,
        n_parallel=args.n_parallel,
    ).run()
