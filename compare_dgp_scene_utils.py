from math import sqrt

import numpy as np
import cv2
import os

# Pixel level diff threshold
PIXEL_DIFF_THRESHOLD = 7
DIFF_PIXEL_COLOUR = [255, 0, 255]


def diff_images(test_image: np.array, target_image: np.array) -> (float, np.array):
    diff, test_image_copy = format_image(test_image, target_image)
    diff_mask = np.where(diff > PIXEL_DIFF_THRESHOLD)
    test_image_copy[diff_mask] = DIFF_PIXEL_COLOUR
    pixel_percent_difference = (len(diff_mask[0]) / diff.size) * 100
    return pixel_percent_difference, test_image_copy


def diff_instance_seg(test_image: np.array, target_image: np.array):
    pass


def format_image(test_image: np.array, target_image: np.array):
    diff = test_image.copy()
    test_image_copy = test_image.copy()
    cv2.absdiff(test_image, target_image, diff)
    if diff.shape[2] != 1:
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    else:
        diff = diff.reshape(diff.shape[0], diff.shape[1])
        test_image_copy = test_image_copy.reshape(test_image_copy.shape[0], test_image_copy.shape[1])
        test_image_copy = cv2.cvtColor(test_image_copy, cv2.COLOR_GRAY2BGR)
    return diff, test_image_copy


def write_image(img: np.array, img_path: str, img_file_name: str):
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, img_path)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    cv2.imwrite(os.path.join(img_path, img_file_name), img)


def calculate_iou(target_box, test_box):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(target_box.x_min, test_box.x_min)
    yA = max(target_box.y_min, test_box.y_min)
    xB = min(target_box.x_max, test_box.x_max)
    yB = min(target_box.y_max, test_box.y_max)
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (target_box.width + 1) * (target_box.height + 1)
    boxBArea = (test_box.width + 1) * (test_box.height + 1)
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
        attribute_errors.append(
            f"The key {key_name} could not be found in test bounding box {test_box.instance_id} but was in target box {target_box.instance_id}"
        )
        return
    if isinstance(target_value, float) and not np.isclose(test_value, target_value) or test_value != target_value:
        attribute_errors.append(
            f"The key {key_name} ({type(test_value).__name__}) is not equal. Test value {test_box_attributes.get(key)}. Target value {target_box_attributes.get(key)}"
        )
        return


def difference_between_vertices(target_box, test_box):
    total_diff = 0
    for i in range(0, 8):
        a = target_box.vertices[i]
        b = test_box.vertices[i]
        total_diff += (
                          sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2) + ((a[2] - b[2]) ** 2))
                      ) * 2  # Punish further distances
    return total_diff


def get_instance_dicts(instance_set_2d):
    instance_ids = np.unique(instance_set_2d.instance_ids.flatten())
    if 0 in instance_ids:
        instance_ids = np.delete(instance_ids, 0)
    instance_dict = dict()
    arr_2d = instance_set_2d.instance_ids.reshape(
        instance_set_2d.instance_ids.shape[0], instance_set_2d.instance_ids.shape[1]
    )
    for inst_id in instance_ids:
        instance_dict[inst_id] = np.where(arr_2d == inst_id)
    return instance_dict, arr_2d


def map_test_to_target(
        test_instances,
        test_instanceseg_2d_arr,
        target_instances,
        target_instanceseg_2d_arr,
        min_instanced_object_percentage_overlap,
):
    test_to_target_map = dict()
    unmatched_test_instances = []
    for test_inst_id, test_mask in test_instances.items():
        target_inst_id = np.bincount(target_instanceseg_2d_arr[test_mask]).argmax()
        if target_inst_id != 0:
            overlap_percent_target = np.bincount(test_instanceseg_2d_arr[target_instances[target_inst_id]])[
                                         test_inst_id
                                     ] / np.sum(np.bincount(test_instanceseg_2d_arr[target_instances[target_inst_id]]))
        else:
            overlap_percent_target = 0
        overlap_percent_test = np.bincount(target_instanceseg_2d_arr[test_mask])[target_inst_id] / np.sum(
            np.bincount(target_instanceseg_2d_arr[test_mask])
        )
        if (
                target_inst_id != 0
                and (overlap_percent_test * 100) >= min_instanced_object_percentage_overlap
                and (overlap_percent_target * 100) >= min_instanced_object_percentage_overlap
        ):
            test_to_target_map[test_inst_id] = target_inst_id
        else:
            unmatched_test_instances.append(test_inst_id)
    unmatched_target_instances = list(set(target_instances.keys()).symmetric_difference(test_to_target_map.values()))

    return test_to_target_map, unmatched_test_instances, unmatched_target_instances
