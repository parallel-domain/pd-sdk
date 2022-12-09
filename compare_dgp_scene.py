import sys
from typing import Tuple

import pytest
import numpy as np
from pathlib import Path

from paralleldomain.decoding.dgp.decoder import DGPDatasetDecoder
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.model.scene import Frame

from compare_dgp_scene_utils import diff_images, write_image, calculate_iou, report_errors, compare_attribute_by_key, \
    difference_between_vertices, get_instance_dicts, map_test_to_target

# Configuration parameters
# Please update the readme if adding / removing any of these parameters
RGB_PIXEL_DIFF_THRESHOLD = 5 # RGB images have more noise
DEPTH_PIXEL_DIFF_THRESHOLD = 2
INST_SEG_PIXEL_DIFF_THRESHOLD = 2
MIN_INSTANCED_OBJECT_PERCENTAGE_OVERLAP = 99.5
SEM_SEG_PIXEL_DIFF_THRESHOLD = 2
MIN_2D_BBOX_BOX_SIZE = 10
MIN_2D_BBOX_IOU = 0.75
MIN_PIXEL_SIZE_FOR_STRICTNESS = 150
MAX_2D_BBOX_PERCENT_SIZE_DIFF = 5
MIN_3D_BBOX_VOLUME = 0.1
MIN_3D_BBOX_BETWEEN_VERTS_DISTANCE = 200
MAX_3D_BBOX_PERCENT_VOLUME_DIFF = 5


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
        sys.stdout.write("\n")
        sys.stdout.write("To generate an HTML test report, add following parameters:\n")
        sys.stdout.write("\t--html=test-results.html --capture=tee-sys\n")
        sys.stdout.write("\n")
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
            # Validate output dir or exit
            path = Path(config.getoption("--output-dir"))
            if path.exists():
                pytest.exit(f"Output dir {path} already exists, please specify a different one")
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
def output_dir(pytestconfig):
    path = Path(pytestconfig.getoption("--output-dir"))
    path.mkdir(parents=True)
    return path


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


def test_camera_rgb(camera_frame_pair, output_dir):
    """RGB data matches for a pair of camera frames"""
    test_camera_frame, target_camera_frame = camera_frame_pair
    test_image, target_image = test_camera_frame.image, target_camera_frame.image
    assert test_image.height == target_image.height
    assert test_image.width == target_image.width

    file_path = f"{str(output_dir / test_camera_frame.sensor_name)}"
    file_name = f"rgb_diff_{test_camera_frame.frame_id}.png"
    pixel_percent_difference, diff_image = diff_images(test_image.rgb,target_image.rgb)

    # TODO we know veg and traffics lights are causing some RGB differences
    # This could be accounted for and allow the diff threshold to be lowered
    if not pixel_percent_difference < RGB_PIXEL_DIFF_THRESHOLD:
        write_image(diff_image, file_path, file_name)
        assert pixel_percent_difference < RGB_PIXEL_DIFF_THRESHOLD

def test_camera_bbox2d(camera_frame_pair):
    general_errors = []
    no_test_box_for_target = []
    no_target_box_for_test = []
    test_matches_two_targets = []
    test_target_attribute_mismatch = []
    test_target_match_different_sizes = []

    """Bbox2D data matches for a pair of camera frames"""
    test_camera_frame, target_camera_frame = camera_frame_pair
    test_bbox2d_boxes = test_camera_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes2D).boxes
    target_bbox2d_boxes = target_camera_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes2D).boxes
    
    """Filter by pixel size"""
    test_bbox2d_boxes = [x for x in test_bbox2d_boxes if x.area > MIN_2D_BBOX_BOX_SIZE]
    target_bbox2d_boxes = [x for x in target_bbox2d_boxes if x.area > MIN_2D_BBOX_BOX_SIZE]

    # Check that we have the same number of bounding boxes
    if len(test_bbox2d_boxes) != len(target_bbox2d_boxes):
        general_errors.append(
            "The length of bounding boxes is not equal. There are {} target boxes while the test has {}".
            format(len(target_bbox2d_boxes), len(test_bbox2d_boxes)))

    #Find best match by IOU
    matched_test_box_instance_ids = set()
    test_instance_id_target_match_pair = dict()
    test_target_match_pair = []
    for target_box in target_bbox2d_boxes:
        best_iou = 0
        best_match = None
        for test_box in test_bbox2d_boxes:
            if (target_box.class_id == test_box.class_id):
                iou = calculate_iou(target_box, test_box)
                if (iou > best_iou):
                    best_iou = iou
                    best_match = test_box
        if target_box.area > MIN_PIXEL_SIZE_FOR_STRICTNESS:
            less_then_minimum_iou = best_iou > MIN_2D_BBOX_IOU
        else:
            less_then_minimum_iou = best_iou > (MIN_2D_BBOX_IOU / 2)
        if best_match == None or target_box.class_id != best_match.class_id or not less_then_minimum_iou:
            no_test_box_for_target.append("Could not find a match for the target bounding box {}, Areas is {}.".format(target_box, target_box.area))
            continue

        # Compare the best match test bbox and target bbox
        best_match_key = str(best_match.instance_id)
        # Check if we have already this test box with another target box
        if test_instance_id_target_match_pair.get(best_match_key, -1) != -1:
            other_target_box = test_instance_id_target_match_pair.get(best_match_key)
            test_matches_two_targets.append(
                "The following test box {} is the cloest match for the two target boxes {} and {}".format(best_match,
                                                                                                          target_box,
                                                                                                          other_target_box))
            continue

        test_instance_id_target_match_pair[best_match_key] = target_box  # Add new pair
        test_target_match_pair.append((best_match, target_box)) # Add to array
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

    """Collect errors where a test - target pair are sizes are not within a threshold"""
    for test_box, target_box in test_target_match_pair:
        percent_diff = abs(test_box.area - target_box.area) / min(test_box.area, target_box.area) * 100
        if min(test_box.area, target_box.area) > MIN_PIXEL_SIZE_FOR_STRICTNESS and percent_diff > MAX_2D_BBOX_PERCENT_SIZE_DIFF:
            test_target_match_different_sizes.append("The test box {} and target box {} have different sizes of {} and {}  with a abs percent diff {}. Class id {}".format(test_box.instance_id, target_box.instance_id, test_box.area, target_box.area, percent_diff, test_box.class_id))


    """Report an errors for every test bounding box that does not have a target pair"""
    for test_box in test_bbox2d_boxes:
        if test_box.instance_id not in matched_test_box_instance_ids:
            no_target_box_for_test.append("Could not find a match for the test bounding box {}. Area is {}.".format(test_box, test_box.area))

    """If a high enough percentage of bounding boxes from test / target are found consider the test passed"""
    all_errors = general_errors + no_test_box_for_target + no_target_box_for_test + test_matches_two_targets + test_target_attribute_mismatch
    if len(all_errors) != 0:
        number_target_boxes = len(target_bbox2d_boxes)
        number_test_boxes = len(test_bbox2d_boxes)
        boxes_matched = len(test_instance_id_target_match_pair)
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
        report_errors("Boxes has size difference outside threshold:", test_target_match_different_sizes)
        report_errors("Following test box matched two target boxes", test_matches_two_targets)
        report_errors("Attribute mismatch box errors:", test_target_attribute_mismatch)

        assert len(all_errors) == 0


def test_camera_bbox3d(camera_frame_pair):

    general_errors = []
    no_test_box_for_target = []
    no_target_box_for_test = []
    test_matches_two_targets = []
    test_target_attribute_mismatch = []
    test_target_match_different_sizes = []

    """Bbox3D data matches for a pair of camera frames"""
    test_camera_frame, target_camera_frame = camera_frame_pair
    test_bbox3d_boxes = test_camera_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D).boxes
    target_bbox3d_boxes = target_camera_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D).boxes

    pre_filter_no_of_test_boxes = len([x for x in test_bbox3d_boxes])
    pre_filter_no_of_target_boxes = len([x for x in target_bbox3d_boxes if x.attributes.get("truncation", -1) != 1])

    """Filter by min volume """
    test_bbox3d_boxes = [x for x in test_bbox3d_boxes if x.volume > MIN_3D_BBOX_VOLUME]
    target_bbox3d_boxes = [x for x in target_bbox3d_boxes if x.volume > MIN_3D_BBOX_VOLUME and x.attributes.get("truncation", -1) != 1]

    # Check that we have the same number of bounding boxes
    number_non_zero_boxes = [x for x in target_bbox3d_boxes if x.num_points != 0]
    if len(test_bbox3d_boxes) != len(number_non_zero_boxes):
        general_errors.append(
            "The length of bounding boxes is not equal. There are {} target boxes while the test has {}".
            format(len(number_non_zero_boxes), len(test_bbox3d_boxes)))

    # Find the closest 3d bound box with the same semantic id, compute the center for each and fine the min distance
    test_instance_id_target_match_pair = dict()
    matched_test_box_instance_ids = set()
    test_target_match_pair = []
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
            no_test_box_for_target.append("Could not find a match for the target bounding box {}, Volume {}.".format(target_box, target_box.volume))
            continue

        # Compare the best match test bbox and target bbox
        best_match_key = best_match.instance_id
        # Check if we have already this test box with another target box
        if test_instance_id_target_match_pair.get(best_match_key, -1) != -1:
            other_target_box = test_instance_id_target_match_pair.get(best_match_key)
            test_matches_two_targets.append(
                "The following test box {} is the closest match for the two target boxes {} and {}".format(best_match,
                                                                                                          target_box,
                                                                                                          other_target_box))
            continue

        test_instance_id_target_match_pair[best_match_key] = target_box  # Add new pair
        test_target_match_pair.append((best_match, target_box))  # Add to array

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


    """Collect errors where a test - target pair are sizes are not within a threshold"""
    for test_box, target_box in test_target_match_pair:
        percent_diff = abs(test_box.volume - target_box.volume) / min(test_box.volume, target_box.volume) * 100
        if percent_diff > MAX_3D_BBOX_PERCENT_VOLUME_DIFF:
            test_target_match_different_sizes.append("The test box {} and target box {} have different sizes of {} and {} with a abs percent diff {}. Class id {}".format(test_box.instance_id, target_box.instance_id, test_box.volume, target_box.volume, percent_diff, test_box.class_id))


    """Report an errors for every test bounding box that does not have a target pair"""
    for test_box in test_bbox3d_boxes:
        if test_box.instance_id not in matched_test_box_instance_ids:
            no_target_box_for_test.append("Could not find a match for the test bounding box {}. Volume is {}.".format(test_box, test_box.volume))

    """If a high enough percentage of bounding boxes from test / target are found consider the test passed"""
    all_errors = general_errors + no_test_box_for_target + no_target_box_for_test + test_matches_two_targets + test_target_attribute_mismatch
    if len(all_errors) != 0:
        number_target_boxes = len(target_bbox3d_boxes)
        number_test_boxes = len(test_bbox3d_boxes)
        boxes_matched = len(test_instance_id_target_match_pair)
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
        report_errors("Boxes has size difference outside threshold:", test_target_match_different_sizes)
        report_errors("Following test box matched two target boxes", test_matches_two_targets)
        report_errors("Attribute mismatch box errors:", test_target_attribute_mismatch)

        assert len(all_errors) == 0


def test_camera_semseg2d(camera_frame_pair, output_dir):
    """Semantic segmentation 2D data matches for a pair of camera frames"""
    test_camera_frame, target_camera_frame = camera_frame_pair
    test_semseg2d = test_camera_frame.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
    target_semseg2d = target_camera_frame.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)
    class_diff = np.setdiff1d(np.unique(test_semseg2d.class_ids.flatten()),
                              np.unique(target_semseg2d.class_ids.flatten()))

    file_path = f"{str(output_dir / test_camera_frame.sensor_name)}"
    file_name = f"semantic_diff_{test_camera_frame.frame_id}.png"
    pixel_percent_difference, diff_image = diff_images(test_semseg2d.rgb_encoded, target_semseg2d.rgb_encoded)

    if not pixel_percent_difference < SEM_SEG_PIXEL_DIFF_THRESHOLD:
        write_image(diff_image, file_path, file_name)
        assert pixel_percent_difference < SEM_SEG_PIXEL_DIFF_THRESHOLD


def test_camera_instanceseg2d(camera_frame_pair, output_dir):
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
                                                                                                  target_instanceseg_2d_arr,
                                                                                                  MIN_INSTANCED_OBJECT_PERCENTAGE_OVERLAP)
    # Highlight pixel in target but not test
    test_image_copy = np.zeros(test_instanceseg2d.rgb_encoded.shape)
    for inst in unmatched_target_instances:
        test_image_copy[target_instances[inst]] = [255, 0, 255]
    for inst in unmatched_test_instances:
        test_image_copy[test_instances[inst]] = [0, 255, 0]

    not_matched_test_pixels = sum([len(test_instances[inst_id][0]) for inst_id in unmatched_test_instances])
    not_matched_target_pixels = sum([len(target_instances[inst_id][0]) for inst_id in unmatched_target_instances])
    instanced_target_pixels = len(np.nonzero(target_instanceseg2d.instance_ids)[0])
    instanced_test_pixels = len(np.nonzero(test_instanceseg2d.instance_ids)[0])
    percentage_image_diff = max(not_matched_test_pixels / instanced_test_pixels, not_matched_target_pixels /instanced_target_pixels) * 100

    if not percentage_image_diff < INST_SEG_PIXEL_DIFF_THRESHOLD:
        image_path, image_name = f"{str(output_dir / test_camera_frame.sensor_name)}", f"instance_diff_{test_camera_frame.frame_id}.png"
        write_image(test_image_copy, image_path, image_name)
        assert percentage_image_diff < INST_SEG_PIXEL_DIFF_THRESHOLD


def test_camera_depth(camera_frame_pair, output_dir):
    """Depth data matches for a pair of camera frames"""
    test_camera_frame, target_camera_frame = camera_frame_pair
    test_depth = test_camera_frame.get_annotations(annotation_type=AnnotationTypes.Depth)
    target_depth = target_camera_frame.get_annotations(annotation_type=AnnotationTypes.Depth)

    file_path = f"{str(output_dir / test_camera_frame.sensor_name)}"
    file_name = f"depth_diff_{test_camera_frame.frame_id}.png"
    pixel_percent_difference, diff_image = diff_images(test_depth.depth, target_depth.depth)

    if not pixel_percent_difference < DEPTH_PIXEL_DIFF_THRESHOLD:
        write_image(diff_image, file_path, file_name)
        assert pixel_percent_difference < DEPTH_PIXEL_DIFF_THRESHOLD


def cli():
    pytest.main(args=sys.argv, plugins=[Conf()])


if __name__ == '__main__':
    cli()
