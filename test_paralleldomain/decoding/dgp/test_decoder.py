import os
from typing import Tuple

import pytest

from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.model.annotation import AnnotationTypes
from test_paralleldomain.decoding.constants import DGP_DATASET_PATH_ENV, DGP_V1_DATASET_PATH_ENV


@pytest.fixture(
    params=[
        ("dgp", DGP_DATASET_PATH_ENV),
        ("dgpv1", DGP_V1_DATASET_PATH_ENV),
    ]
)
def dataset_format_and_path(request) -> Tuple[str, str]:
    dataset_format, path_env = request.param
    if path_env not in os.environ:
        pytest.skip()
    return dataset_format, os.environ[path_env]


def test_flip_semseg_and_instance_seg_files_with_custom_annotation_map(dataset_format_and_path: Tuple[str, str]):
    dataset_format, path = dataset_format_and_path

    custom_annotation_type_map = {
        "0": AnnotationTypes.BoundingBoxes2D,
        "1": AnnotationTypes.BoundingBoxes3D,
        "3": AnnotationTypes.SemanticSegmentation3D,
        "5": AnnotationTypes.InstanceSegmentation3D,
        "6": AnnotationTypes.Depth,
        "7": AnnotationTypes.SurfaceNormals3D,
        "9": AnnotationTypes.SceneFlow,
        "10": AnnotationTypes.SurfaceNormals2D,
        "8": AnnotationTypes.OpticalFlow,
        "12": AnnotationTypes.Albedo2D,
        "13": AnnotationTypes.MaterialProperties2D,
        "15": AnnotationTypes.MaterialProperties3D,
        # flip instance and semseg
        "4": AnnotationTypes.SemanticSegmentation2D,
        "2": AnnotationTypes.InstanceSegmentation2D,
    }

    dataset = decode_dataset(
        dataset_path=path, dataset_format=dataset_format, custom_annotation_type_map=custom_annotation_type_map
    )
    assert dataset is not None

    for camera_frame in dataset.camera_frames:
        semseg = camera_frame.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
        max_semseg_class_id = max(camera_frame.class_maps[AnnotationTypes.SemanticSegmentation2D].class_ids)
        max_isnstance_id = semseg.class_ids.max()
        assert max_isnstance_id > max_semseg_class_id
        break
