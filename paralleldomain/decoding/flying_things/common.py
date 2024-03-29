from datetime import datetime, timedelta
from typing import List, Set, Tuple

from paralleldomain.model.type_aliases import FrameId, SceneName
from paralleldomain.utilities.any_path import AnyPath

CLEAN_IMAGE_FOLDER_1_NAME = "frames_cleanpass"
CLEAN_IMAGE_FOLDER_2_NAME = "image_clean"
FINAL_IMAGE_FOLDER_1_NAME = "frames_finalpass"
FINAL_IMAGE_FOLDER_2_NAME = "image_final"
OPTICAL_FLOW_FOLDER_NAME = "optical_flow"
OPTICAL_FLOW_FORWARD_DIRECTION_NAME = "into_future"
OPTICAL_FLOW_BACKWARD_DIRECTION_NAME = "into_past"

LEFT_SENSOR_NAME = "left"
RIGHT_SENSOR_NAME = "right"

SPLIT_NAME_TO_FOLDER_NAME = {
    "training": "TRAIN",
    "train": "TRAIN",
    "TRAIN": "TRAIN",
    "testing": "TEST",
    "test": "TEST",
    "TEST": "TEST",
    "validation": "val",
    "val": "val",
    "VAL": "val",
}


def frame_id_to_timestamp(frame_id: str) -> datetime:
    """
    frame_id is of the form "xxxxx_imgx.p"
    Since there is no true framerate or timestamp in FlyingThings, we make one up.
    """
    epoch_time = datetime(1970, 1, 1)
    seconds = int(frame_id) + 0.1
    timestamp = epoch_time + timedelta(seconds)
    return timestamp


def get_frame_ids_of_subset_scene(
    split_list: List[int],
    scene_name: SceneName,
) -> List[FrameId]:
    _, list_index = scene_name.split("/", maxsplit=1)
    list_index = int(list_index)
    total_prev_ids = sum(split_list[:list_index])
    num_ids = split_list[list_index]
    return [str(total_prev_ids + i).zfill(7) for i in range(num_ids)]


def get_scene_folder(
    dataset_path: AnyPath,
    scene_name: SceneName,
    split_name: str,
) -> AnyPath:
    frame_pass, scene_name = scene_name.split("/", maxsplit=1)
    return dataset_path / frame_pass / split_name / scene_name


def get_image_folder_name(scene_name: SceneName) -> str:
    frame_pass, scene_name = scene_name.split("/", maxsplit=1)
    return {
        CLEAN_IMAGE_FOLDER_1_NAME: CLEAN_IMAGE_FOLDER_2_NAME,
        FINAL_IMAGE_FOLDER_1_NAME: FINAL_IMAGE_FOLDER_2_NAME,
    }[frame_pass]


def get_scene_flow_folder(
    dataset_path: AnyPath,
    scene_name: SceneName,
    split_name: str,
) -> AnyPath:
    _, scene_name = scene_name.split("/", maxsplit=1)
    return dataset_path / OPTICAL_FLOW_FOLDER_NAME / split_name / scene_name


def decode_frame_id_set(
    scene_name: SceneName,
    is_full_dataset_format: bool,
    split_name: str,
    dataset_path: AnyPath,
    split_list: List[int],
    sensor_name: str,
) -> Set[FrameId]:
    if is_full_dataset_format:
        folder_path = (
            get_scene_folder(dataset_path=dataset_path, scene_name=scene_name, split_name=split_name)
            / get_image_folder_name(scene_name=scene_name)
            / sensor_name
        )
        frame_ids = {img.split(".png")[0] for img in folder_path.glob("*.png")}
        return frame_ids
    else:
        frame_ids = get_frame_ids_of_subset_scene(scene_name=scene_name, split_list=split_list)
        return set(frame_ids)
