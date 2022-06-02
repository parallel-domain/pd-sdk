import os
from tempfile import TemporaryDirectory
from typing import List

import pytest

from paralleldomain.common.dgp.v1.constants import DirectoryName
from paralleldomain.decoding.dgp.v1.decoder import DGPDatasetDecoder
from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.encoding.dgp.v1.dataset import DGPDatasetEncoder
from paralleldomain.encoding.dgp.v1.pipeline_encoder import DGPV1DatasetPipelineEncoder
from paralleldomain.model.annotation import Annotation
from paralleldomain.model.dataset import Dataset
from paralleldomain.model.sensor import CameraSensorFrame, FilePathedDataType, LidarSensorFrame
from paralleldomain.utilities.any_path import AnyPath


def check_files_in_folder(path: AnyPath, expected_num: int, expected_suffix: str = ".bin", check_suffix: bool = True):
    assert path.exists()
    files = list(path.iterdir())
    assert len(files) == expected_num
    if check_suffix:
        assert all([str(f).endswith(expected_suffix) for f in files])


def check_if_files_are_binary(
    output_path: AnyPath,
    dataset: Dataset,
    scene_names: List[str],
    camera_names: List[str],
    lidar_names: List[str],
    num_frames: int,
    num_calibration_files: int = 1,
):
    assert (output_path / "scene_dataset.bin").exists()
    expected_suffix = ".bin"
    for scene_name in scene_names:
        scene_folder = output_path / scene_name
        assert scene_folder.exists()
        assert any(
            [
                True
                for f in (output_path / scene_name).iterdir()
                if f.name.endswith(".bin") and f.name.startswith("scene_")
            ]
        )

        check_files_in_folder(
            path=scene_folder / DirectoryName.CALIBRATION,
            expected_num=num_calibration_files,
            expected_suffix=expected_suffix,
        )
        check_files_in_folder(
            path=scene_folder / DirectoryName.ONTOLOGY, expected_num=1, expected_suffix=expected_suffix
        )

        for sensor_name in camera_names:
            check_files_in_folder(
                path=scene_folder / DirectoryName.BOUNDING_BOX_2D / sensor_name,
                expected_num=num_frames,
                expected_suffix=expected_suffix,
            )
            check_files_in_folder(
                path=scene_folder / DirectoryName.BOUNDING_BOX_3D / sensor_name,
                expected_num=num_frames,
                expected_suffix=expected_suffix,
            )
            check_files_in_folder(
                path=scene_folder / DirectoryName.DEPTH / sensor_name, expected_num=num_frames, expected_suffix=".npz"
            )
            check_files_in_folder(
                path=scene_folder / DirectoryName.SEMANTIC_SEGMENTATION_2D / sensor_name,
                expected_num=num_frames,
                expected_suffix=".png",
            )
            check_files_in_folder(
                path=scene_folder / DirectoryName.INSTANCE_SEGMENTATION_2D / sensor_name,
                expected_num=num_frames,
                expected_suffix=".png",
            )
            check_files_in_folder(
                path=scene_folder / DirectoryName.RGB / sensor_name, expected_num=num_frames, expected_suffix=".png"
            )
            check_files_in_folder(
                path=scene_folder / DirectoryName.MOTION_VECTORS_2D / sensor_name,
                expected_num=num_frames,
                expected_suffix=".png",
            )

        for sensor_name in lidar_names:
            check_files_in_folder(
                path=scene_folder / DirectoryName.BOUNDING_BOX_3D / sensor_name,
                expected_num=num_frames,
                expected_suffix=expected_suffix,
            )


def test_encode_to_binary_proto(dataset: Dataset):
    if "SKIP_ENCODER" not in os.environ:
        with TemporaryDirectory() as temp_dir:
            num_frames = 3
            output_path = AnyPath(temp_dir)

            encoder = DGPV1DatasetPipelineEncoder.from_path(
                dataset_path=dataset.path,
                dataset_format=dataset.format,
                decoder_kwargs=dataset.decoder_init_kwargs,
                output_path=output_path,
                set_stop=2,
                workers=4,
                max_in_queue_size=10,
                fs_copy=False,
                allowed_frames=[str(i) for i in range(num_frames)],
                copy_all_available_sensors_and_annotations=True,
                encode_to_binary=True,
            )
            encoder.encode_dataset()
            # scene = dataset.get_scene(scene_name=dataset.scene_names[0])
            check_if_files_are_binary(
                output_path=output_path,
                dataset=dataset,
                scene_names=dataset.scene_names[:2],
                camera_names=list(dataset.camera_names),
                num_frames=num_frames,
                lidar_names=list(dataset.lidar_names),
            )


def test_encode_and_read_binary_proto(dataset: Dataset):
    if "SKIP_ENCODER" not in os.environ:
        with TemporaryDirectory() as temp_dir:
            num_frames = 3
            output_path = AnyPath(temp_dir)

            encoder = DGPV1DatasetPipelineEncoder.from_path(
                dataset_path=dataset.path,
                dataset_format=dataset.format,
                decoder_kwargs=dataset.decoder_init_kwargs,
                output_path=output_path,
                set_stop=2,
                workers=4,
                max_in_queue_size=10,
                fs_copy=False,
                allowed_frames=[str(i) for i in range(num_frames)],
                copy_all_available_sensors_and_annotations=True,
                encode_to_binary=True,
            )
            encoder.encode_dataset()
            # scene = dataset.get_scene(scene_name=dataset.scene_names[0])

            binary_dataset = decode_dataset(dataset_path=output_path, dataset_format="dgpv1")

            for sensor_frame in binary_dataset.sensor_frames:
                if isinstance(sensor_frame, CameraSensorFrame):
                    img = sensor_frame.image.rgb
                    assert img is not None
                elif isinstance(sensor_frame, LidarSensorFrame):
                    xyz = sensor_frame.point_cloud.xyz
                    assert xyz is not None
                for annotation_type in sensor_frame.available_annotation_types:
                    annotation = sensor_frame.get_annotations(annotation_type=annotation_type)
                    assert annotation is not None
                    assert isinstance(annotation, Annotation)
