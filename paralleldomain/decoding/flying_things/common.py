import re
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np

from paralleldomain.model.type_aliases import SceneName
from paralleldomain.utilities.any_path import AnyPath

CLEAN_IMAGE_FOLDER_1_NAME = "frames_cleanpass"
CLEAN_IMAGE_FOLDER_2_NAME = "image_clean"
FINAL_IMAGE_FOLDER_1_NAME = "frames_finalpass"
FINAL_IMAGE_FOLDER_2_NAME = "image_final"
OPTICAL_FLOW_FOLDER_NAME = "optical_flow"

LEFT_SENSOR_NAME = "left"
RIGHT_SENSOR_NAME = "right"

SPLIT_NAME_TO_FOLDER_NAME = {"training": "TRAIN", "train": "TRAIN", "testing": "TEST", "test": "TEST"}


def frame_id_to_timestamp(frame_id: str) -> datetime:
    """
    frame_id is of the form "xxxxx_imgx.p"
    Since there is no true framerate or timestamp in FlyingChairs, we make one up.
    """
    epoch_time = datetime(1970, 1, 1)
    seconds = int(frame_id) + 0.1
    timestamp = epoch_time + timedelta(seconds)
    return timestamp


def get_scene_folder(
    dataset_path: AnyPath,
    scene_name: SceneName,
    split_name: str,
) -> AnyPath:
    frame_pass, scene_name = scene_name.split("/")
    return dataset_path / frame_pass / split_name / scene_name


def get_scene_flow_folder(
    dataset_path: AnyPath,
    scene_name: SceneName,
    split_name: str,
) -> AnyPath:
    _, scene_name = scene_name.split("/")
    return dataset_path / OPTICAL_FLOW_FOLDER_NAME / split_name / scene_name


def read_pfm(file_path: AnyPath) -> Tuple[np.ndarray, float]:
    """
    Adapted from: https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py
    """
    with file_path.open("rb") as file:
        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file.")

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:  # little-endian
            endian = "<"
            scale = -scale
        else:
            endian = ">"  # big-endian

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data, scale


def read_flow(file_path: AnyPath) -> np.ndarray:
    """
    Adapted from: https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py
    """
    name = file_path.name
    if name.endswith(".pfm") or name.endswith(".PFM"):
        return read_pfm(file_path=file_path)[0][:, :, 0:2]

    with file_path.open("rb") as f:
        header = f.read(4)
        if header.decode("utf-8") != "PIEH":
            raise Exception("Flow file header does not contain PIEH")

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()
        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    return flow.astype(np.float32)
