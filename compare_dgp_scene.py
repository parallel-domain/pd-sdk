import sys
from typing import Tuple

import pytest
import numpy as np

from paralleldomain.decoding.dgp.decoder import DGPDatasetDecoder
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.model.scene import Frame


class Conf:
    """
    FIXME: load this plugin when executing with pytest
           for now we can only run with python
    """
    def __init__(self):
        self.test_decoder = None
        self.target_decoder = None
        self.test_dataset = None
        self.target_dataset = None
        self.test_scene = None
        self.target_scene = None

    def print_help_prologue(self):
        sys.stdout.write("This script compares two DGP datasets and reports differences between them.\n")
        sys.stdout.write("\n")
        sys.stdout.write("Example command:\n")
        sys.stdout.write("\tpython compare_dgp_scene.py --test-dataset /path/to/test/dataset --target-dataset /path/to/target/dataset --test-scene scene_000000 -v\n")
        sys.stdout.write("\n")
        sys.stdout.write("The tests are named according to their annotation, frame and sensor as follows:\n")
        sys.stdout.write("\ttest_<annotation>[f<frame#>-<sensor_name>]\n")
        sys.stdout.write("\n")
        sys.stdout.write("We can use standard pytest keyword filtering to select filter tests along these dimension.\n")
        sys.stdout.write("\tRun only the test for a specific annotation on a single frame and single sensor:\n")
        sys.stdout.write("\t\t-k test_camera_depth[f8-Rear]\n")
        sys.stdout.write("\tRun all tests for a single annotation:\n")
        sys.stdout.write("\t\t-k test_camera_depth\n")
        sys.stdout.write("\tRun all tests for a specific frame:\n")
        sys.stdout.write("\t\t-k [f8\n")
        sys.stdout.write("\tRun all tests for a specific sensor:\n")
        sys.stdout.write("\t\t-k Rear]\n")
        sys.exit(0)


    def pytest_addoption(self, parser):
        parser.addoption("--test-dataset", action="store", help="Dataset under test")
        parser.addoption("--target-dataset", action="store", help="Dataset to use for verification")
        parser.addoption("--test-scene", action="store", help="Name of scene in test dataset")
        parser.addoption("--target-scene", action="store", help="Name of scene in target dataset")

    def pytest_configure(self, config):
        if not config.getoption("--help"):
            test_dataset_path = config.getoption("--test-dataset")
            target_dataset_path = config.getoption("--target-dataset")
            test_scene_name = config.getoption("--test-scene")
            target_scene_name = config.getoption("--target-scene")
            if not test_dataset_path:
                raise Exception("--test-dataset option is required")
            if not target_dataset_path:
                raise Exception("--target-dataset option is required")
            if not test_scene_name:
                raise Exception("--test-scene option is required")
            target_scene_name = target_scene_name or test_scene_name

            self.test_decoder = DGPDatasetDecoder(dataset_path=test_dataset_path)
            self.target_decoder = DGPDatasetDecoder(dataset_path=target_dataset_path)
            self.test_dataset = self.test_decoder.get_dataset()
            self.target_dataset = self.target_decoder.get_dataset()
            self.test_scene = self.test_dataset.get_scene(test_scene_name)
            self.target_scene = self.target_dataset.get_scene(target_scene_name)
        else:
            self.print_help_prologue()

    def pytest_generate_tests(self, metafunc):
        # Let's parameterize the fixture `frame_pair` by all frames in the datasets
        if 'frame_pair' in metafunc.fixturenames:
            num_frames_test_scene = len(self.test_scene.frames)
            num_frames_target_scene = len(self.target_scene.frames)
            if num_frames_test_scene != num_frames_target_scene:
                raise Exception(f"Mismatch in # frames between test scene ({num_frames_test_scene}) "
                                f"and target scene ({num_frames_target_scene})")
            metafunc.parametrize(
                'frame_pair',
                zip(self.test_scene.frames, self.target_scene.frames),
                ids=[f"f{f.frame_id}" for f in self.test_scene.frames]
            )
        # Let's parameterize the fixture `camera_name` by all camera names in the datasets
        if 'camera_name' in metafunc.fixturenames:
            test_camera_names = self.test_scene.camera_names
            target_camera_names = self.target_scene.camera_names
            if test_camera_names != target_camera_names:
                raise Exception(f"Mismatch in test camera names ({test_camera_names}) "
                                f"and target camera names ({target_camera_names})")
            metafunc.parametrize('camera_name', test_camera_names)
        # Let's parameterize the fixture `scene_pair`
        if 'scene_pair' in metafunc.fixturenames:
            metafunc.parametrize('scene_pair', [(self.test_scene, self.target_scene)], ids=['scene'])


@pytest.fixture(scope='session')
def scene_pair(request) -> Tuple[Frame, Frame]:
    """
    Returns scene pair (test_scene, target_scene)
    """
    return request.param


@pytest.fixture(scope='session')
def frame_pair(request) -> Tuple[Frame, Frame]:
    """
    Returns a pair of (test, target) frames that should be compared
    Parameterized to provide all frames
    """
    return request.param


@pytest.fixture(scope='session')
def camera_name(request) -> str:
    """
    Returns a camera name
    Parameterized to provide all camera names
    """
    return request.param


@pytest.fixture
def camera_frame_pair(frame_pair, camera_name) -> Tuple[CameraSensorFrame, CameraSensorFrame]:
    """
    Returns a pair of camera frames (test, target)
    Parameterized to provide all possible frame pairs across all cameras
    """
    test_frame, target_frame = frame_pair
    return test_frame.get_camera(camera_name=camera_name), target_frame.get_camera(camera_name=camera_name)


def test_ontology(scene_pair):
    """Ontologies match between the scenes"""
    test_scene, target_scene = scene_pair

    test_pass = True

    if set(test_scene.available_annotation_types) != set(target_scene.available_annotation_types):
        print(f"Available annotation types mismatch:\n {test_scene.available_annotation_types}\n\n {target_scene.available_annotation_types}\n")
        # Don't fail on these

    common_annotation_types = set(test_scene.available_annotation_types).intersection(target_scene.available_annotation_types)
    for annotation in common_annotation_types:
        test_class_map, target_class_map = test_scene.get_class_map(annotation), target_scene.get_class_map(annotation)
        test_class_dict = {k: (v.name, v.id, v.instanced) for k, v in test_class_map.items()}
        target_class_dict = {k: (v.name, v.id, v.instanced) for k, v in target_class_map.items()}
        if test_class_dict != target_class_dict:
            print(f"Ontology for {annotation} mismatch:\n {test_class_dict}\n\n {target_class_dict}\n")
            test_pass = False

    assert test_pass


def test_camera_model(camera_frame_pair):
    """Camera model matches for a pair of camera frames"""
    test_camera_frame, target_camera_frame = camera_frame_pair
    test_extrinsics, target_extrinsics = test_camera_frame.extrinsic, target_camera_frame.extrinsic
    test_intrinsics, target_intrinsics = test_camera_frame.intrinsic, target_camera_frame.intrinsic

    test_pass = True

    if not np.allclose(test_extrinsics.transformation_matrix, target_extrinsics.transformation_matrix):
        print(f"Extrinsics mismatch:\n {test_extrinsics}\n\n {target_extrinsics}\n")
        test_pass = False

    if not np.allclose(test_intrinsics.camera_matrix, target_intrinsics.camera_matrix):
        print(f"Camera matrix mismatch:\n {test_intrinsics.camera_matrix}\n\n {target_intrinsics.camera_matrix}\n")
        test_pass = False

    if not np.allclose(test_intrinsics.distortion_parameters, target_intrinsics.distortion_parameters):
        print(f"Camera distortion mismatch\n {test_intrinsics.distortion_parameters}\n\n {target_intrinsics.distortion_parameters}\n")
        test_pass = False

    assert test_pass


def test_camera_rgb(camera_frame_pair):
    """RGB data matches for a pair of camera frames"""
    test_camera_frame, target_camera_frame = camera_frame_pair
    test_image, target_image = test_camera_frame.image, target_camera_frame.image
    assert test_image.height == target_image.height
    assert test_image.width == target_image.width


def test_camera_bbox2d(camera_frame_pair):
    """Bbox2D data matches for a pair of camera frames"""
    test_camera_frame, target_camera_frame = camera_frame_pair
    test_bbox2d = test_camera_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes2D)
    target_bbox2d = target_camera_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes2D)
    assert len(test_bbox2d.boxes) == len(target_bbox2d.boxes)


def test_camera_bbox3d(camera_frame_pair):
    """Bbox3D data matches for a pair of camera frames"""
    test_camera_frame, target_camera_frame = camera_frame_pair
    test_bbox3d = test_camera_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D)
    target_bbox3d = target_camera_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D)
    assert len(test_bbox3d.boxes) == len(target_bbox3d.boxes)


def test_camera_semseg2d(camera_frame_pair):
    """Semantic segmentation 2D data matches for a pair of camera frames"""
    test_camera_frame, target_camera_frame = camera_frame_pair
    test_semseg2d = test_camera_frame.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
    target_semseg2d = target_camera_frame.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
    assert np.array_equal(test_semseg2d.class_ids, target_semseg2d.class_ids)


def test_camera_instanceseg2d(camera_frame_pair):
    """Instance segmentation 2D data matches for a pair of camera frames"""
    test_camera_frame, target_camera_frame = camera_frame_pair
    test_instanceseg2d = test_camera_frame.get_annotations(annotation_type=AnnotationTypes.InstanceSegmentation2D)
    target_instanceseg2d = target_camera_frame.get_annotations(annotation_type=AnnotationTypes.InstanceSegmentation2D)
    assert np.array_equal(test_instanceseg2d.instance_ids, target_instanceseg2d.instance_ids)


def test_camera_depth(camera_frame_pair):
    """Depth data matches for a pair of camera frames"""
    test_camera_frame, target_camera_frame = camera_frame_pair
    test_depth = test_camera_frame.get_annotations(annotation_type=AnnotationTypes.Depth)
    target_depth = target_camera_frame.get_annotations(annotation_type=AnnotationTypes.Depth)
    assert np.array_equal(test_depth.depth, target_depth.depth)


def cli():
    pytest.main(args=sys.argv, plugins=[Conf()])


if __name__ == '__main__':
    cli()
