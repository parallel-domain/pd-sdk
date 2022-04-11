import os
from tempfile import TemporaryDirectory
from typing import List

import pytest

from paralleldomain.common.dgp.v1.constants import DirectoryName
from paralleldomain.decoding.dgp.v1.decoder import DGPDatasetDecoder
from paralleldomain.encoding.dgp.v1.dataset import DGPDatasetEncoder
from paralleldomain.encoding.dgp.v1.pipeline_encoder import DGPV1DatasetPipelineEncoder
from paralleldomain.model.dataset import Dataset
from paralleldomain.model.sensor import FilePathedDataType
from paralleldomain.utilities.any_path import AnyPath


def check_if_files_exist(
    output_path: AnyPath,
    dataset: Dataset,
    scene_names: List[str],
    camera_names: List[str],
    lidar_names: List[str],
    num_frames: int,
    num_calibration_files: int = 1,
):
    assert (output_path / "scene_dataset.json").exists()
    for scene_name in scene_names:
        scene_folder = output_path / scene_name
        assert scene_folder.exists()
        assert any(
            [
                True
                for f in (output_path / scene_name).iterdir()
                if f.name.endswith(".json") and f.name.startswith("scene_")
            ]
        )
        assert (scene_folder / DirectoryName.CALIBRATION).exists()
        assert len(list((scene_folder / DirectoryName.CALIBRATION).iterdir())) == num_calibration_files
        assert (scene_folder / DirectoryName.ONTOLOGY).exists()
        assert len(list((scene_folder / DirectoryName.ONTOLOGY).iterdir())) == 1
        for sensor_name in camera_names:
            assert (scene_folder / DirectoryName.BOUNDING_BOX_2D / sensor_name).exists()
            assert len(list((scene_folder / DirectoryName.BOUNDING_BOX_2D / sensor_name).iterdir())) == num_frames
            assert (scene_folder / DirectoryName.DEPTH / sensor_name).exists()
            assert len(list((scene_folder / DirectoryName.DEPTH / sensor_name).iterdir())) == num_frames
            assert (scene_folder / DirectoryName.SEMANTIC_SEGMENTATION_2D / sensor_name).exists()
            assert (
                len(list((scene_folder / DirectoryName.SEMANTIC_SEGMENTATION_2D / sensor_name).iterdir())) == num_frames
            )
            assert (scene_folder / DirectoryName.INSTANCE_SEGMENTATION_2D / sensor_name).exists()
            assert (
                len(list((scene_folder / DirectoryName.INSTANCE_SEGMENTATION_2D / sensor_name).iterdir())) == num_frames
            )
            assert (scene_folder / DirectoryName.RGB / sensor_name).exists()
            assert len(list((scene_folder / DirectoryName.RGB / sensor_name).iterdir())) == num_frames
            assert (scene_folder / DirectoryName.MOTION_VECTORS_2D / sensor_name).exists()
            assert len(list((scene_folder / DirectoryName.MOTION_VECTORS_2D / sensor_name).iterdir())) == num_frames
            assert (scene_folder / DirectoryName.BOUNDING_BOX_3D / sensor_name).exists()
            assert len(list((scene_folder / DirectoryName.BOUNDING_BOX_3D / sensor_name).iterdir())) == num_frames

        for sensor_name in lidar_names:
            assert (scene_folder / DirectoryName.BOUNDING_BOX_3D / sensor_name).exists()
            assert len(list((scene_folder / DirectoryName.BOUNDING_BOX_3D / sensor_name).iterdir())) == num_frames


def test_encoding_of_modified_scene(dataset: Dataset):
    if "SKIP_ENCODER" not in os.environ:
        with TemporaryDirectory() as temp_dir:
            num_frames = 3
            output_path = AnyPath(temp_dir)
            annotation_types = dataset.available_annotation_types
            annotation_types = list(
                set(annotation_types).intersection(
                    {
                        FilePathedDataType.BoundingBoxes2D,
                        FilePathedDataType.SemanticSegmentation2D,
                        FilePathedDataType.Depth,
                        FilePathedDataType.OpticalFlow,
                        FilePathedDataType.BoundingBoxes3D,
                        FilePathedDataType.InstanceSegmentation2D,
                    }
                )
            )
            if len(dataset.camera_names) > 0:
                annotation_types.append(FilePathedDataType.Image)
            if len(dataset.lidar_names) > 0:
                annotation_types.append(FilePathedDataType.PointCloud)

            encoder = DGPV1DatasetPipelineEncoder.from_path(
                dataset_path=dataset.path,
                dataset_format=dataset.format,
                decoder_kwargs=dataset.decoder_init_kwargs,
                output_path=output_path,
                # sensor_names=[n for n in dataset.camera_names],
                set_stop=2,
                workers=1,
                max_in_queue_size=10,
                fs_copy=False,
                allowed_frames=[str(i) for i in range(num_frames)],
                copy_data_types=annotation_types,
            )
            encoder.encode_dataset()
            # scene = dataset.get_scene(scene_name=dataset.scene_names[0])
            check_if_files_exist(
                output_path=output_path,
                dataset=dataset,
                scene_names=dataset.scene_names[:2],
                camera_names=list(dataset.camera_names),
                num_frames=num_frames,
                lidar_names=list(dataset.lidar_names),
            )


def test_should_copy_callback(dataset: Dataset):
    if "SKIP_ENCODER" not in os.environ:
        with TemporaryDirectory() as temp_dir:
            num_frames = 3
            output_path = AnyPath(temp_dir)
            only_copy_sensor_name = list(dataset.camera_names)[0]
            encoder = DGPV1DatasetPipelineEncoder.from_path(
                dataset_path=dataset.path,
                dataset_format=dataset.format,
                decoder_kwargs=dataset.decoder_init_kwargs,
                output_path=output_path,
                set_stop=2,
                workers=1,
                max_in_queue_size=10,
                fs_copy=False,
                allowed_frames=[str(i) for i in range(num_frames)],
                copy_data_types=[
                    FilePathedDataType.BoundingBoxes2D,
                    FilePathedDataType.Image,
                    FilePathedDataType.SemanticSegmentation2D,
                    FilePathedDataType.Depth,
                    FilePathedDataType.OpticalFlow,
                    FilePathedDataType.BoundingBoxes3D,
                    FilePathedDataType.InstanceSegmentation2D,
                ],
                should_copy_callbacks={
                    FilePathedDataType.Image: lambda d, s: s.sensor_name == only_copy_sensor_name,
                },
            )
            with pytest.raises(KeyError, match="rgb"):
                encoder.encode_dataset()


def test_encoding_of_inplace_scene(dataset: Dataset):
    if "SKIP_ENCODER" not in os.environ:
        with TemporaryDirectory() as temp_dir:
            num_frames = 3
            output_path = AnyPath(temp_dir)

            copy_data_types = dataset.available_annotation_types
            copy_data_types = list(
                set(copy_data_types).intersection(
                    {
                        FilePathedDataType.BoundingBoxes2D,
                        FilePathedDataType.SemanticSegmentation2D,
                        FilePathedDataType.Depth,
                        FilePathedDataType.OpticalFlow,
                        FilePathedDataType.BoundingBoxes3D,
                        FilePathedDataType.InstanceSegmentation2D,
                    }
                )
            )
            if len(dataset.camera_names) > 0:
                copy_data_types.append(FilePathedDataType.Image)
            if len(dataset.lidar_names) > 0:
                copy_data_types.append(FilePathedDataType.PointCloud)

            encoder = DGPV1DatasetPipelineEncoder.from_path(
                dataset_path=dataset.path,
                dataset_format=dataset.format,
                decoder_kwargs=dataset.decoder_init_kwargs,
                output_path=output_path,
                set_stop=2,
                fs_copy=True,
                allowed_frames=[str(i) for i in range(num_frames)],
                copy_data_types=copy_data_types,
            )
            encoder.encode_dataset()
            scene = dataset.get_scene(scene_name=dataset.scene_names[0])
            check_if_files_exist(
                output_path=output_path,
                dataset=dataset,
                scene_names=dataset.scene_names,
                camera_names=scene.camera_names,
                num_frames=num_frames,
                lidar_names=[],
            )

            encoder = DGPV1DatasetPipelineEncoder.from_path(
                dataset_path=output_path,
                inplace=True,
                dataset_format=dataset.format,
                decoder_kwargs=dataset.decoder_init_kwargs,
                output_path=output_path,
                sensor_names={"virtual_cam": "camera_front"},
                set_stop=2,
                fs_copy=True,
                allowed_frames=[str(i) for i in range(num_frames)],
                copy_data_types=copy_data_types,
            )
            encoder.encode_dataset()
            scene = dataset.get_scene(scene_name=dataset.scene_names[0])
            check_if_files_exist(
                output_path=output_path,
                dataset=dataset,
                scene_names=dataset.scene_names,
                camera_names=scene.camera_names + ["virtual_cam"],
                num_frames=num_frames,
                num_calibration_files=2,  # old one may still exist but is not referenced
                lidar_names=[],
            )
