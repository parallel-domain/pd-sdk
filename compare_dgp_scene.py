import sys
from enum import Enum
from typing import Tuple

import pytest
import numpy as np
from math import sqrt

from paralleldomain.decoding.dgp.decoder import DGPDatasetDecoder
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.model.scene import Frame

from utils import diff_images, write_image

RGB_PIXEL_DIFF_THRESHOLD = 5
DEPTH_PIXEL_DIFF_THRESHOLD = 5
INST_SEG_PIXEL_DIFF_THRESHOLD = 5
MIN_INSTANCED_OBJECT_PERCENTAGE_OVERLAP = 99.9
SEM_SEG_PIXEL_DIFF_THRESHOLD = 5
MIN_2D_BBOX_BOX_SIZE = 110
MIN_2D_BBOX_IOU = 0.75
# The current 3d bbox noise ratio is quite high for small volumes
MIN_3D_BBOX_VOLUME = 2  # Increasing to 2 hugely reduces errors but reduced accuracy
MIN_3D_BBOX_BETWEEN_VERTS_DISTANCE = 200
OUTPUT_DIRECTORY = ""


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
        sys.stdout.write(
            "\tpython compare_dgp_scene.py --test-dataset /path/to/test/dataset --target-dataset /path/to/target/dataset --test-scene scene_000000 --output-dir 'test_results' -v\n")
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
        parser.addoption("--output-dir", action="store", help="Output directory for test results and images")

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
            if not config.getoption("--output-dir"):
                raise Exception("--output-dir option is required")
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

@pytest.fixture(autouse=True, scope='session')
def cmd_param(pytestconfig):
    path = pytestconfig.getoption("--output-dir")
    global OUTPUT_DIRECTORY
    OUTPUT_DIRECTORY = path


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
    pixel_percent_difference = diff_images(test_image.rgb,
                                           target_image.rgb,
                                           f"{OUTPUT_DIRECTORY}{test_camera_frame.sensor_name}",
                                           f"rgb_diff_{test_camera_frame.frame_id}.png",
                                           save_images=True)
    assert pixel_percent_difference < RGB_PIXEL_DIFF_THRESHOLD
    assert test_image.height == target_image.height
    assert test_image.width == target_image.width


def test_camera_bbox2d(camera_frame_pair):
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
    test_bbox2d_boxes = [x for x in test_bbox2d_boxes if x.area > MIN_2D_BBOX_BOX_SIZE]
    target_bbox2d_boxes = [x for x in target_bbox2d_boxes if x.area > MIN_2D_BBOX_BOX_SIZE]

    # Check that we have the same number of bounding boxes
    if len(test_bbox2d_boxes) != len(target_bbox2d_boxes):
        # TODO migrate to error class
        general_errors.append(
            "The length of bounding boxes is not equal. There are {} target boxes while the test has {}".
            format(len(target_bbox2d_boxes), len(test_bbox2d_boxes)))

    #Find best match by IOU
    matched_test_box_instance_ids = set()
    test_instnace_id_target_match_pair = dict()
    for target_box in target_bbox2d_boxes:
        best_iou = 0
        best_match = None
        for test_box in test_bbox2d_boxes:
            if (target_box.class_id == test_box.class_id):
                iou = calculate_iou(target_box,test_box)
                if (iou > best_iou):
                    best_iou = iou
                    best_match = test_box
        if best_match == None or target_box.class_id != best_match.class_id or best_iou < MIN_2D_BBOX_IOU:
            no_test_box_for_target.append("Could not find a match for the target bounding box {}, Areas is {}".format(target_box, target_box.area))
            continue

        # Compare the best match test bbox and target bbox
        best_match_key = str(best_match.instance_id)
        # Check if we have already this test box with another target box
        if test_instnace_id_target_match_pair.get(best_match_key, -1) != -1:
            other_target_box = test_instnace_id_target_match_pair.get(best_match_key)
            test_matches_two_targets.append(
                "The following test box {} is the cloest match for the two target boxes {} and {}".format(best_match,
                                                                                                          target_box,
                                                                                                          other_target_box))
            continue

        test_instnace_id_target_match_pair[best_match_key] = target_box  # Add new pair
        matched_test_box_instance_ids.add(best_match.instance_id)

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
            test_target_attribute_mismatch.append("The test bounding box {} and target bounding box {} had the following attribute errors:\n\t{}".format(best_match,target_box, "\n\t".join(attribute_errors)))

    """Report an errors for every test bounding box that does not have a target pair"""
    for test_box in test_bbox2d_boxes:
        if test_box.instance_id not in matched_test_box_instance_ids:
            no_target_box_for_test.append("Could not find a match for the test bounding box {}. Area is {}. Attributes {}".format(test_box, test_box.area, test_box.attributes))

    """If a high enough percentage of bounding boxes from test / target are found consider the test passed"""
    all_errors = general_errors + no_test_box_for_target + no_target_box_for_test + test_matches_two_targets + test_target_attribute_mismatch
    if len(all_errors) != 0:
        number_target_boxes = len(target_bbox2d_boxes)
        number_test_boxes = len(test_bbox2d_boxes)
        boxes_matched = len(test_instnace_id_target_match_pair)
        percentage_test_boxes_matched = boxes_matched / number_test_boxes * 100
        percentage_target_boxes_matched = boxes_matched / number_target_boxes * 100
        percentage_of_matched_boxes_with_attribute_mismatch = boxes_matched / max(number_test_boxes, number_target_boxes) * 100

        # Stats
        print("There are {} general errors.\n"
              "There are {} test boxes. {:3.2f}% could be matched\n"
              "There are {} target boxes. {:3.2f}% could be matched\n"
              "There are {} test boxes that matched two target boxes\n"
              "Out of {} matched boxes. {:3.2f}% have attribute mismatches\n"
              .format(len(general_errors), number_test_boxes, percentage_test_boxes_matched, number_target_boxes,
                      percentage_target_boxes_matched, len(test_matches_two_targets), boxes_matched, percentage_of_matched_boxes_with_attribute_mismatch))

        # Report errors
        report_errors("General errors:", general_errors)
        report_errors("No test box test for target box errors:", no_test_box_for_target)
        report_errors("No target box for test box errors:", no_target_box_for_test)
        report_errors("Following test box matched two target boxes", test_matches_two_targets)
        report_errors("Attribute mismatch box errors:", test_target_attribute_mismatch)

        assert False


def calculate_iou(target_box, test_box):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(target_box.x_min, test_box.x_min)
    yA = max(target_box.y_min, test_box.y_min)
    xB = min(target_box.x_max, test_box.x_max)
    yB = min(target_box.y_max, test_box.y_max)
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = target_box.area
    boxBArea = test_box.area
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def report_errors(error_section_message, errors):
    if len(errors) == 0:
        return
    print()
    print(error_section_message)
    for i in errors:
        print(i)


def compare_attribute_by_key(test_box, target_box, key, is_in_user_data, attribute_errors):
    test_box_attributes = test_box.attributes
    target_box_attributes = target_box.attributes
    key_name = key
    if is_in_user_data:
        test_box_attributes = test_box.attributes["user_data"]
        target_box_attributes = target_box.attributes["user_data"]
        key_name = "user_data/" + key

    # Check if exists in both
    test_value = test_box_attributes.get(key)
    target_value = target_box_attributes.get(key)
    if target_value is None:
        return
    if test_value is None:
        attribute_errors.append(f"The key {key_name} could not be found in test bounding box {test_box}")
        return
    if isinstance(target_value, float) and not np.isclose(test_value, target_value) or \
            test_value != target_value:
        attribute_errors.append(
            f"The key {key_name} ({type(test_value).__name__}) is not equal. Test value {test_box_attributes.get(key)}. Target value {target_box_attributes.get(key)}")
        return


def difference_between_vertices(target_box, test_box):
    total_diff = 0
    for i in range(0, 8):
        a = target_box.vertices[i]
        b = test_box.vertices[i]
        for n in range(0, 3):
            total_diff += sqrt((a[n] - b[n]) ** 2)
    return total_diff

def test_camera_bbox3d(camera_frame_pair):

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

    pre_filter_no_of_test_boxes = len([x for x in test_bbox3d_boxes if x.attributes.get("truncation", -1) != 1])
    pre_filter_no_of_target_boxes = len([x for x in target_bbox3d_boxes if x.attributes.get("truncation", -1) != 1])

    """Filter by min volume """
    test_bbox3d_boxes = [x for x in test_bbox3d_boxes if x.volume > MIN_3D_BBOX_VOLUME and x.attributes.get("truncation", -1) != 1]
    target_bbox3d_boxes = [x for x in target_bbox3d_boxes if x.volume > MIN_3D_BBOX_VOLUME and x.attributes.get("truncation", -1) != 1]

    # Check that we have the same number of bounding boxes
    number_non_zero_boxes = [x for x in target_bbox3d_boxes if x.num_points != 0]
    if len(test_bbox3d_boxes) != len(number_non_zero_boxes):
        # TODO migrate to error class
        general_errors.append(
            "The length of bounding boxes is not equal. There are {} target boxes while the test has {}".
            format(len(number_non_zero_boxes), len(test_bbox3d_boxes)))

    # Find the closest 3d bound box with the same semantic id, compute the center for each and fine the min distance
    test_target_match_pair = dict()
    matched_test_box_instance_ids = set()
    for target_box in target_bbox3d_boxes:
        best_match = None
        min_distance = 1000
        for test_box in test_bbox3d_boxes:
            if (test_box.class_id != target_box.class_id):
                continue
            d = difference_between_vertices(target_box, test_box)

            if (d < min_distance  and d < MIN_3D_BBOX_BETWEEN_VERTS_DISTANCE):
                min_distance = d
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
        matched_test_box_instance_ids.add(best_match.instance_id)
        # Compare all sorted attributes
        attribute_errors = []
        compare_attribute_by_key(best_match, target_box, "brake_light", False, attribute_errors)
        compare_attribute_by_key(best_match, target_box, "left_indicator", False, attribute_errors)
        compare_attribute_by_key(best_match, target_box, "left_indicator", False, attribute_errors)
        compare_attribute_by_key(best_match, target_box, "parked_vehicle", False, attribute_errors)
        compare_attribute_by_key(best_match, target_box, "occlusion", False, attribute_errors)

        if (len(attribute_errors) != 0):
            test_target_attribute_mismatch.append("The test bounding box {} and target bounding box {} had the following attribute errors:\n\t{}".format(best_match,target_box, "\n\t".join(attribute_errors)))

    """Report an errors for every test bounding box that does not have a target pair"""
    for test_box in test_bbox3d_boxes:
        if test_box.instance_id not in matched_test_box_instance_ids:
            no_target_box_for_test.append("Could not find a match for the test bounding box {}. Volume is {}. Attributes {}".format(test_box, test_box.volume, test_box.attributes))

    """If a high enough percentage of bounding boxes from test / target are found consider the test passed"""
    all_errors = general_errors + no_test_box_for_target + no_target_box_for_test + test_matches_two_targets + test_target_attribute_mismatch
    if len(all_errors) != 0:
        number_target_boxes = len(target_bbox3d_boxes)
        number_test_boxes = len(test_bbox3d_boxes)
        boxes_matched = len(test_target_match_pair)
        percentage_test_boxes_matched = boxes_matched / number_test_boxes * 100
        percentage_target_boxes_matched = boxes_matched / number_target_boxes * 100
        percentage_of_matched_boxes_with_attribute_mismatch = boxes_matched / max(number_test_boxes,
                                                                                  number_target_boxes) * 100
        # Stats
        print("There was {} test and {} target boxes before filtering".format(pre_filter_no_of_test_boxes,pre_filter_no_of_target_boxes))
        print("There are {} general errors.\n"
              "There are {} test boxes. {:3.2f}% could be matched\n"
              "There are {} target boxes. {:3.2f}% could be matched\n"
              "There are {} test boxes that matched two target boxes\n"
              "Out of {} matched boxes. {:3.2f}% have attribute mismatches\n"
              .format(len(general_errors), number_test_boxes, percentage_test_boxes_matched, number_target_boxes,
                      percentage_target_boxes_matched, len(test_matches_two_targets), boxes_matched,
                      percentage_of_matched_boxes_with_attribute_mismatch))

        # Error messages
        report_errors("General errors:", general_errors)
        report_errors("No test box test for target box errors:", no_test_box_for_target)
        report_errors("No target box for test box errors:", no_target_box_for_test)
        report_errors("Following test box matched two target boxes", test_matches_two_targets)
        report_errors("Attribute mismatch box errors:", test_target_attribute_mismatch)

        assert False


def test_camera_semseg2d(camera_frame_pair):
    """Semantic segmentation 2D data matches for a pair of camera frames"""
    test_camera_frame, target_camera_frame = camera_frame_pair
    test_semseg2d = test_camera_frame.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
    target_semseg2d = target_camera_frame.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
    pixel_percent_difference = diff_images(test_semseg2d.rgb_encoded, target_semseg2d.rgb_encoded,
                                           f"{OUTPUT_DIRECTORY}{test_camera_frame.sensor_name}",
                                           f"semantic_diff_{test_camera_frame.frame_id}.png",
                                           save_images=True)
    class_diff = np.setdiff1d(np.unique(test_semseg2d.class_ids.flatten()),
                              np.unique(target_semseg2d.class_ids.flatten()))
    assert pixel_percent_difference < SEM_SEG_PIXEL_DIFF_THRESHOLD
    assert np.array_equal(test_semseg2d.class_ids, target_semseg2d.class_ids)


def convert_rgb_to_number(rgb):
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    return (red << 16) + (green << 8) + blue


def get_instance_dicts(instance_set_2d):
    instance_ids = np.unique(instance_set_2d.instance_ids.flatten())
    if 0 in instance_ids:
        instance_ids = np.delete(instance_ids, 0)
    instance_dict = dict()
    arr_2d = instance_set_2d.instance_ids.reshape(instance_set_2d.instance_ids.shape[0],
                                                  instance_set_2d.instance_ids.shape[1])
    for inst_id in instance_ids:
        instance_dict[inst_id] = np.where(arr_2d == inst_id)
    return instance_dict, arr_2d


def map_test_to_target(test_instances, test_instanceseg_2d_arr, target_instances, target_instanceseg_2d_arr):
    test_to_target_map = dict()
    unmatched_test_instances = []
    for test_inst_id, test_mask in test_instances.items():
        target_inst_id = np.bincount(target_instanceseg_2d_arr[test_mask]).argmax()
        if target_inst_id != 0:
            overlap_percent_target = \
                np.bincount(test_instanceseg_2d_arr[target_instances[target_inst_id]])[
                    test_inst_id] / np.sum(
                    np.bincount(test_instanceseg_2d_arr[target_instances[target_inst_id]]))
        else:
            overlap_percent_target = 0
        overlap_percent_test = np.bincount(target_instanceseg_2d_arr[test_mask])[target_inst_id] / np.sum(
            np.bincount(target_instanceseg_2d_arr[test_mask]))
        if target_inst_id != 0 and (overlap_percent_test * 100) >= MIN_INSTANCED_OBJECT_PERCENTAGE_OVERLAP \
                and (overlap_percent_target * 100) >= MIN_INSTANCED_OBJECT_PERCENTAGE_OVERLAP:
            test_to_target_map[test_inst_id] = target_inst_id
        else:
            unmatched_test_instances.append(test_inst_id)
    unmatched_target_instances = list(set(target_instances.keys()).symmetric_difference(test_to_target_map.values()))

    return test_to_target_map, unmatched_test_instances, unmatched_target_instances


def test_camera_instanceseg2d(camera_frame_pair):
    """Instance segmentation 2D data matches for a pair of camera frames"""
    test_camera_frame, target_camera_frame = camera_frame_pair
    test_instanceseg2d = test_camera_frame.get_annotations(annotation_type=AnnotationTypes.InstanceSegmentation2D)
    target_instanceseg2d = target_camera_frame.get_annotations(annotation_type=AnnotationTypes.InstanceSegmentation2D)

    # Store the instance mask at each instance id
    target_instances, target_instanceseg_2d_arr = get_instance_dicts(target_instanceseg2d)
    test_instances, test_instanceseg_2d_arr = get_instance_dicts(test_instanceseg2d)

    # map test instance ids to target instance ids that line up and track unmatched test instance ids
    test_to_target_map, unmatched_test_instances, unmatched_target_instances = map_test_to_target(test_instances,
                                                                                                  test_instanceseg_2d_arr,
                                                                                                  target_instances,
                                                                                                  target_instanceseg_2d_arr)

    test_image_copy = np.zeros(test_instanceseg2d.rgb_encoded.shape)
    for inst in unmatched_target_instances:
        test_image_copy[target_instances[inst]] = [255, 0, 255]
    for inst in unmatched_test_instances:
        test_image_copy[test_instances[inst]] = [0, 255, 0]

    not_matched_test_pixels = sum([len(test_instances[inst_id][0]) for inst_id in unmatched_test_instances])
    not_matched_target_pixels = sum([len(target_instances[inst_id][0]) for inst_id in unmatched_target_instances])
    instanced_target_pixels = len(np.nonzero(target_instanceseg2d.instance_ids)[0])
    instanced_test_pixels = len(np.nonzero(test_instanceseg2d.instance_ids)[0])
    percentage_image_diff = max(not_matched_test_pixels, not_matched_target_pixels) / min(instanced_target_pixels,
                                                                                          instanced_test_pixels) * 100
    if True:
        image_path, image_name = f"{OUTPUT_DIRECTORY}\\{test_camera_frame.sensor_name}", f"instance_diff_{test_camera_frame.frame_id}.png"
        write_image(test_image_copy, image_path, image_name)

    assert percentage_image_diff < INST_SEG_PIXEL_DIFF_THRESHOLD

# def test_camera_instanceseg2d(camera_frame_pair):
#     """Instance segmentation 2D data matches for a pair of camera frames"""
#     test_camera_frame, target_camera_frame = camera_frame_pair
#     test_instanceseg2d = test_camera_frame.get_annotations(annotation_type=AnnotationTypes.InstanceSegmentation2D)
#     target_instanceseg2d = target_camera_frame.get_annotations(annotation_type=AnnotationTypes.InstanceSegmentation2D)
#
#     # TODO this need to be speed up
#     # Build a set of pixels in each instance id
#     target_instances = dict()
#     target_rgb = target_instanceseg2d.rgb_encoded
#     instanced_target_pixels = 0
#     for instance_id in range(0, len(target_rgb)):
#         for j in range(0,len(target_rgb[instance_id])):
#             target_instance_id = convert_rgb_to_number(target_rgb[instance_id][j])
#             if target_instance_id != 0:
#                 instance_set = target_instances.get(target_instance_id, set())
#                 instance_set.add((instance_id,j))
#                 target_instances[target_instance_id] = instance_set
#                 instanced_target_pixels += 1
#     test_instances = dict()
#     test_rgb = test_instanceseg2d.rgb_encoded
#     instanced_text_pixels = 0
#     for instance_id in range(0, len(test_rgb)):
#         for j in range(0,len(test_rgb[instance_id])):
#             test_instance_id = convert_rgb_to_number(test_rgb[instance_id][j])
#             if test_instance_id != 0:
#                 instance_set = test_instances.get(test_instance_id, set())
#                 instance_set.add((instance_id,j))
#                 test_instances[test_instance_id] = instance_set
#                 instanced_text_pixels += 1
#
#     # Try and match each target instance to a test instance set
#     matched_test_instance = set()
#     not_match_target_instance = set()
#     for target_instance_key in target_instances.keys():
#         best_match = 0
#         best_test_match_key = None
#         for test_instance_key in test_instances.keys():
#             test_set = test_instances.get(test_instance_key)
#             target_set = target_instances.get(target_instance_key)
#             match_percentage = len(target_set.intersection(test_set))/ len(target_set) * 100
#             if match_percentage > best_match and match_percentage > MIN_INSTANCED_OBJECT_PERCENTAGE_OVERLAP:
#                 best_match = match_percentage
#                 best_test_match_key = test_instance_key
#         if best_test_match_key != None:
#             matched_test_instance.add(best_test_match_key)
#         else:
#             not_match_target_instance.add(target_instance_key)
#     not_matched_test_instanced = [x for x in test_instances.keys() if x not in matched_test_instance]
#
#     # Create diff image from two overlays
#     diff_rgb = np.zeros((target_rgb.shape), dtype=int)
#     not_matched_test_pixels = 0
#     not_matched_target_pixels = 0
#     for instance_id in not_match_target_instance:
#         target_set = target_instances.get(instance_id)
#         for pixel_location in target_set:
#             diff_rgb[pixel_location[0],pixel_location[1]] = [255, 0, 255]
#             not_matched_target_pixels += 1
#     for instance_id in not_matched_test_instanced:
#         test_set = test_instances.get(instance_id)
#         for pixel_location in test_set:
#             diff_rgb[pixel_location[0],pixel_location[1]] = [0, 255, 0]
#             not_matched_test_pixels += 1
#
#     percentage_image_diff = max(not_matched_test_pixels, not_matched_target_pixels) / min(instanced_target_pixels, instanced_text_pixels) * 100
#     if (True):
#         image_path, image_name = f"{OUTPUT_DIRECTORY}{test_camera_frame.sensor_name}", f"instance_diff_{test_camera_frame.frame_id}.png"
#         write_image(diff_rgb, image_path, image_name)
#
#     assert percentage_image_diff < INST_SEG_PIXEL_DIFF_THRESHOLD

def test_camera_depth(camera_frame_pair):
    """Depth data matches for a pair of camera frames"""
    test_camera_frame, target_camera_frame = camera_frame_pair
    test_depth = test_camera_frame.get_annotations(annotation_type=AnnotationTypes.Depth)
    target_depth = target_camera_frame.get_annotations(annotation_type=AnnotationTypes.Depth)
    pixel_percent_difference = diff_images(test_depth.depth, target_depth.depth,
                                           f"{OUTPUT_DIRECTORY}{test_camera_frame.sensor_name}",
                                           f"depth_diff_{test_camera_frame.frame_id}.png",
                                           save_images=True)
    assert pixel_percent_difference < DEPTH_PIXEL_DIFF_THRESHOLD
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
