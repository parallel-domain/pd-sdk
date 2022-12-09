import numpy as np
import cv2
import os

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
