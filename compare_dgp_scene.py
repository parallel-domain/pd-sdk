import copy
import sys
from enum import Enum
from typing import Tuple
import statistics

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

    # TODO This is running every test twice
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
                zip(self.test_scene.frames[0:1], self.target_scene.frames[0:1]),
                ids=[f"f{f.frame_id}" for f in self.test_scene.frames[0:1]]
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
    """Max difference allowed between boxes """
    max_x_y_pixel_difference = 15  # Define how the size of the pixel box used to search for a matching box
    max_percentage_size_difference = 5  # Define max percentage difference in size # TODO change to min union percentage
    min_box_size = 500
    min_h_w = 5

    # TODO improve error reporting // Should we collect errors by type?
    general_errors = []
    no_test_box_for_target = []
    no_target_box_for_test = []
    test_matches_two_targets = []
    test_target_attribute_mismatch = []

    """Bbox2D data matches for a pair of camera frames"""
    test_camera_frame, target_camera_frame = camera_frame_pair
    test_bbox2d_boxes = test_camera_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes2D).boxes
    target_bbox2d_boxes = target_camera_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes2D).boxes
    
    """Filter by pixel size"""
    test_bbox2d_boxes = [x for x in test_bbox2d_boxes if x.area > min_box_size and x.width > min_h_w and x.height > min_h_w]
    target_bbox2d_boxes = [x for x in target_bbox2d_boxes if x.area > min_box_size and x.width > min_h_w and x.height > min_h_w]

    # Check that we have the same number of bounding boxes
    if len(test_bbox2d_boxes) != len(target_bbox2d_boxes):
        # TODO migrate to error class
        general_errors.append(
            "The length of bounding boxes is not equal. There are {} target boxes while the test has {}".
            format(len(target_bbox2d_boxes), len(test_bbox2d_boxes)))

    """ Sort all test bounding boxes. Store by x,y tuple for key. We then use the target boxes to do a look up
    # If we find more than one boxes find the best match in terms of size
    # Then perform a deep comparison of the boxes"""
    test_boxes_by_x_y = dict()
    for test_box in test_bbox2d_boxes:
        test_boxes_by_x_y[(test_box.x, test_box.y)] = test_box

    # For each target bounding box try and match it to a test
    test_target_match_pair = dict()
    for target_box in target_bbox2d_boxes:
        # Try to find any boxes that match with the max_x_y_pixel_difference range
        found_boxes = locate_2d_bounding_boxes_by_xy(target_box, test_boxes_by_x_y, max_x_y_pixel_difference)
        if len(found_boxes) == 0:
            no_test_box_for_target.append("Could not find a match for the target bounding box {}, Areas is ".format(target_box, target_box.area))
            continue
        best_match = find_closest_box_by_size_and_class_id(target_box, found_boxes, max_percentage_size_difference)
        if best_match == None:
            no_test_box_for_target.append("Could not find a match for the target bounding box {}, Areas is ".format(target_box, target_box.area))
            continue

        # Compare the best match test bbox and target bbox
        best_match_key = (best_match.x, best_match.y)
        # Check if we have already this test box with another target box
        if test_target_match_pair.get(best_match_key, -1) != -1:
            other_target_box = test_target_match_pair.get(best_match_key)
            test_matches_two_targets.append(
                "The following test box {} is the cloest match for the two target boxes {} and {}".format(best_match,
                                                                                                          target_box,
                                                                                                          other_target_box))
            continue

        test_target_match_pair[best_match_key] = target_box  # Add new pair

        # Compare all sorted attributes
        attribute_errors = []
        compare_attribute_by_key(best_match, target_box, "brake_light", False, attribute_errors)
        compare_attribute_by_key(best_match, target_box, "left_indicator", False, attribute_errors)
        compare_attribute_by_key(best_match, target_box, "left_indicator", False, attribute_errors)
        compare_attribute_by_key(best_match, target_box, "parked_vehicle", False, attribute_errors)

        compare_attribute_by_key(best_match, target_box, "trailer_angle", True, attribute_errors)
        compare_attribute_by_key(best_match, target_box, "truncation", True, attribute_errors)
        compare_attribute_by_key(best_match, target_box, "visibility", True, attribute_errors)

        if (len(attribute_errors) != 0):
            test_target_attribute_mismatch.append("The test bounding box {} and target bounding box {} had the following attribute errors {}".format(best_match,target_box, attribute_errors))

    """Report an errors for every test bounding box that does not have a target pair"""
    # copy dict then remove all keys that are in test_matched_boxes
    unmatched_test_boxes = test_boxes_by_x_y.copy()
    for key in test_target_match_pair.keys():
        unmatched_test_boxes.pop(key)
    for box in unmatched_test_boxes.values():
        if box.area != 0:
            no_target_box_for_test.append("Could not find a match for the test bounding box {}".format(box))

    data = []
    for i in test_bbox2d_boxes:
        data.append(i.area)
    data.sort()
    data = data[1:-2] # remove first and last item
    mean = np.mean(data)
    std = np.std(data)

    """If a high enough percentage of bounding boxes from test / target are found consider the test passed"""
    number_target_boxes = len(target_bbox2d_boxes)
    number_test_boxes = len(test_bbox2d_boxes)
    boxes_matched = len(test_target_match_pair)
    percentage_test_boxes_matched = boxes_matched / number_test_boxes * 100
    percentage_target_boxes_matched = boxes_matched / number_target_boxes * 100
    percentage_of_matched_boxes_with_attribute_mismatch = boxes_matched / max(number_test_boxes, number_target_boxes) * 100

    print("There are {} general errors.\n"
          "There are {} test boxes. {:3.2f}% could be matched\n"
          "There are {} target boxes. {:3.2f}% could be matched\n"
          "Out of {} matched boxes. {:3.2f}% have attribute mismatches\n"
          .format(len(general_errors), number_test_boxes, percentage_test_boxes_matched, number_target_boxes,
                  percentage_target_boxes_matched, boxes_matched, percentage_of_matched_boxes_with_attribute_mismatch))


    all_errors = general_errors + no_test_box_for_target + no_target_box_for_test + test_matches_two_targets + test_target_attribute_mismatch
    if len(all_errors) != 0:
        print("General errors:")
        for i in general_errors:
            print(i)
        print("\nMissing test box errors:")
        for i in no_test_box_for_target:
            print(i)
        print("\nMissing target box errors:")
        for i in no_target_box_for_test:
            print(i)
        print("\nAttribute mismatch box errors:")
        for i in test_target_attribute_mismatch:
            print(i)
        print("\nTest box matches two targets boxes errors:")
        for i in test_matches_two_targets:
            print(i)

        assert False

def compare_attribute_by_key(test_box, target_box, key, is_in_user_data, attribute_errors):
    test_box_attributes = test_box.attributes
    target_box_attributes = target_box.attributes
    key_name = key
    if (is_in_user_data):
        test_box_attributes = test_box.attributes["user_data"]
        target_box_attributes = target_box.attributes["user_data"]
        key_name = "user_data/" + key

    # Check if exists in both
    if (test_box_attributes.get(key, -1) == -1 and target_box_attributes.get(key,-1) == -1):
        return # attribute is not relevant for this type
    if (target_box_attributes.get(key,-1) == -1):
        return # Key is not in target so ignore for now
    if (test_box_attributes.get(key, -1) == -1):
        attribute_errors.append("The key {} could not be found in test bounding box".format(key_name, test_box))
        return
    if (test_box_attributes.get(key) != target_box_attributes.get(key)):
        attribute_errors.append("The key {} is not equal. Test key value {}. Target key value {}".format(key_name, test_box, target_box, test_box_attributes.get(key), target_box_attributes.get(key)))
        return



# TODO migrate to union / intersection instead
def find_closest_box_by_size_and_class_id(target_box, found_boxes, max_percentage_size_difference):
    # At least one box found in the required pixel range
    # Find best match and perform a deep comparison
    best_match = None
    best_percentage_match_difference = 100
    target_box_size = target_box.area
    for test_box in found_boxes:
        percentage_difference = abs(target_box_size - test_box.area) / target_box_size * 100
        # Find the bbox that is the closest match in terms of size
        if (target_box.class_id == test_box.class_id
                and (percentage_difference <= max_percentage_size_difference)
                and percentage_difference < best_percentage_match_difference):
            best_percentage_match_difference = percentage_difference
            best_match = test_box
    return best_match


def locate_2d_bounding_boxes_by_xy(target_box, test_boxes_by_x_y, max_x_y_pixel_difference):
    found_boxes = []
    for x in range(-max_x_y_pixel_difference, max_x_y_pixel_difference + 1):  # Search 5 pixels either side
        for y in range(-max_x_y_pixel_difference, max_x_y_pixel_difference + 1):
            coordiantes = (target_box.x + x, target_box.y + y)
            matched_box = test_boxes_by_x_y.get(coordiantes, -1)
            if (matched_box != -1):
                found_boxes.append(copy.deepcopy(matched_box))
    return found_boxes


def test_camera_bbox3d(camera_frame_pair):
    # Set high level diff variables
    max_translation_distance = 10  # Define how the size of the pixel box used to search for a matching box
    max_percentage_size_difference = 10  # Define max percentage difference in size
    min_volume = 1

    # TODO improve error reporting // Should we collect errors by type?
    general_errors = []
    no_test_box_for_target = []
    no_target_box_for_test = []
    test_matches_two_targets = []
    test_target_attribute_mismatch = []

    """Bbox3D data matches for a pair of camera frames"""
    test_camera_frame, target_camera_frame = camera_frame_pair
    test_bbox3d_boxes = test_camera_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D).boxes
    target_bbox3d_boxes = target_camera_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D).boxes

    """Filter by min volume """
    test_bbox3d_boxes = [x for x in test_bbox3d_boxes if x.volume > min_volume and x.attributes.get("truncation", -1) != 1]
    target_bbox3d_boxes = [x for x in target_bbox3d_boxes if x.volume > min_volume and x.attributes.get("truncation", -1) != 1]

    # Check that we have the same number of bounding boxes
    number_non_zero_boxes = [x for x in target_bbox3d_boxes if x.num_points != 0]
    if len(test_bbox3d_boxes) != len(number_non_zero_boxes):
        # TODO migrate to error class
        general_errors.append(
            "The length of bounding boxes is not equal. There are {} target boxes while the test has {}".
            format(len(number_non_zero_boxes), len(test_bbox3d_boxes)))

    # Find the closest 3d bound box with the same semantic id and compare
    test_target_match_pair = dict()
    for target_box in target_bbox3d_boxes:
        best_match = None
        min_distance = 1000
        target_box_area = target_box.width * target_box.height * target_box.length
        for test_box in test_bbox3d_boxes:
            if (target_box.class_id != test_box.class_id):
                continue
            translation_distance = abs(target_box.pose.translation[0] - test_box.pose.translation[0]) + abs(target_box.pose.translation[1] - test_box.pose.translation[1]) + abs(target_box.pose.translation[2] - test_box.pose.translation[2])
            test_box_area = test_box.width * test_box.height * test_box.length
            percentage_area_diff = abs(test_box_area - target_box_area) / test_box_area * 100
            if (translation_distance < max_translation_distance and translation_distance< min_distance and percentage_area_diff < max_percentage_size_difference):
                min_distance = translation_distance
                best_match = test_box

        if best_match == None:
            no_test_box_for_target.append("Could not find a match for the target bounding box {}, Volume {}".format(target_box, target_box.volume))
            continue

        # Compare the best match test bbox and target bbox
        best_match_key = best_match.instance_id
        # Check if we have already this test box with another target box
        if test_target_match_pair.get(best_match_key, -1) != -1:
            other_target_box = test_target_match_pair.get(best_match_key)
            test_matches_two_targets.append(
                "The following test box {} is the closest match for the two target boxes {} and {}".format(best_match,
                                                                                                          target_box,
                                                                                                          other_target_box))
            continue

        test_target_match_pair[best_match_key] = target_box  # Add new pair
        # Compare all sorted attributes
        attribute_errors = []
        compare_attribute_by_key(best_match, target_box, "brake_light", False, attribute_errors)
        compare_attribute_by_key(best_match, target_box, "left_indicator", False, attribute_errors)
        compare_attribute_by_key(best_match, target_box, "left_indicator", False, attribute_errors)
        compare_attribute_by_key(best_match, target_box, "parked_vehicle", False, attribute_errors)
        compare_attribute_by_key(best_match, target_box, "occlusion", False, attribute_errors)

        if (len(attribute_errors) != 0):
            test_target_attribute_mismatch.append("The test bounding box {} and target bounding box {} had the following attribute errors {}".format(best_match,target_box, attribute_errors))

    """Report an errors for every test bounding box that does not have a target pair"""
    # copy dict then remove all keys that are in test_matched_boxes
    for test_box in test_bbox3d_boxes:
        if (test_box.instance_id not in test_target_match_pair.keys()):
            no_target_box_for_test.append("Could not find a match for the test bounding box {}, Volume {}".format(test_box, test_box.volume))

    """If a high enough percentage of bounding boxes from test / target are found consider the test passed"""
    number_target_boxes = len(target_bbox3d_boxes)
    number_test_boxes = len(test_bbox3d_boxes)
    boxes_matched = len(test_target_match_pair)
    percentage_test_boxes_matched = boxes_matched / number_test_boxes * 100
    percentage_target_boxes_matched = boxes_matched / number_target_boxes * 100
    percentage_of_matched_boxes_with_attribute_mismatch = boxes_matched / max(number_test_boxes, number_target_boxes) * 100

    print("There are {} general errors.\n"
          "There are {} test boxes. {:3.2f}% could be matched\n"
          "There are {} target boxes. {:3.2f}% could be matched\n"
          "Out of {} matched boxes. {:3.2f}% have attribute mismatches\n"
          .format(len(general_errors), number_test_boxes, percentage_test_boxes_matched, number_target_boxes,
                  percentage_target_boxes_matched, boxes_matched, percentage_of_matched_boxes_with_attribute_mismatch))


    all_errors = general_errors + no_test_box_for_target + no_target_box_for_test + test_matches_two_targets + test_target_attribute_mismatch
    if len(all_errors) != 0:
        print("General errors:")
        for i in general_errors:
            print(i)
        print("\nMissing test box errors:")
        for i in no_test_box_for_target:
            print(i)
        print("\nMissing target box errors:")
        for i in no_target_box_for_test:
            print(i)
        print("\nAttribute mismatch box errors:")
        for i in test_target_attribute_mismatch:
            print(i)
        print("\nTest box matches two targets boxes errors:")
        for i in test_matches_two_targets:
            print(i)

        assert False




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

class BoundingBoxErrorType(Enum):
    GENERAL_ERROR = 1
    TEST_MISSING_FROM_TARGET = 2
    TARGET_MISSING_FROM_TEST = 3
    TEST_MATCHES_MULTIPLE_TARGETS = 4
    TEST_TARGET_ATTRIBUTES_ARE_MISMATCHED = 5

class BoundingBoxCompareError:
    def __int__(self, errorType: BoundingBoxErrorType, errorMessage: str):
        pass

    def generate_error(self) -> str:
        # TODO generate an error based on error types
        pass


def cli():
    pytest.main(args=sys.argv, plugins=[Conf()])


if __name__ == '__main__':
    cli()
