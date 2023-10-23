import json
import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from paralleldomain.data_lab.config.reactor import ReactorObject
from paralleldomain.data_lab.reactor import (
    PerCameraReactorInput,
    decode_rgba_image_png_base64,
    encode_rgba_image_png_base64,
    get_instance_mask_and_prompts,
)
from paralleldomain.model.annotation import InstanceSegmentation2D, SemanticSegmentation2D

HEIGHT = 16
WIDTH = 32


@pytest.fixture()
def target_image() -> np.ndarray:
    return (np.ones((HEIGHT, WIDTH, 3), dtype=int) * 255).astype(np.uint8)


@pytest.fixture()
def target_depth() -> np.ndarray:
    return (np.ones((HEIGHT, WIDTH, 1), dtype=int) * 255).astype(np.float16)


@pytest.fixture()
def target_frame_semseg_mask() -> SemanticSegmentation2D:
    class_ids = np.zeros((HEIGHT, WIDTH, 1), dtype=int)
    class_ids[0, 0] = 22
    class_ids[4, 4] = 22
    class_ids[8, 8] = 10
    class_ids[12, 12] = 22
    return SemanticSegmentation2D(class_ids=class_ids)


@pytest.fixture()
def target_frame_instance_mask() -> InstanceSegmentation2D:
    instance_ids = np.zeros((HEIGHT, WIDTH, 1), dtype=int)
    instance_ids[0, 0] = 1
    instance_ids[4, 4] = 3000
    instance_ids[8, 8] = 1001
    instance_ids[12, 12] = 1002
    return InstanceSegmentation2D(instance_ids=instance_ids)


@pytest.fixture()
def empty_image() -> np.ndarray:
    img = (np.ones((HEIGHT, WIDTH, 3), dtype=int) * 255).astype(np.uint8)
    img[0, 0, 0] = 0
    return img


@pytest.fixture()
def empty_frame_semseg_mask() -> SemanticSegmentation2D:
    class_ids = np.zeros((HEIGHT, WIDTH, 1), dtype=int)
    class_ids[0, 0] = 22
    class_ids[4, 4] = 22
    class_ids[8, 8] = 10
    class_ids[15, 10] = 99
    return SemanticSegmentation2D(class_ids=class_ids)


@pytest.fixture()
def empty_frame_instance_mask() -> InstanceSegmentation2D:
    instance_ids = np.zeros((HEIGHT, WIDTH, 1), dtype=int)
    instance_ids[0, 0] = 1
    instance_ids[4, 4] = 3000
    instance_ids[8, 8] = 1001
    instance_ids[15, 10] = 1002
    return InstanceSegmentation2D(instance_ids=instance_ids)


@pytest.mark.parametrize(
    "replacement_probability",
    [1.0, 0.0],
)
def test_get_instance_masks_and_prompts_from_class_id(
    target_image: np.ndarray,
    target_depth: np.ndarray,
    target_frame_semseg_mask: SemanticSegmentation2D,
    target_frame_instance_mask: InstanceSegmentation2D,
    replacement_probability: float,
):
    camera_reactor_input = PerCameraReactorInput(
        target_image=target_image,
        target_depth=target_depth,
        target_frame_semseg_mask=target_frame_semseg_mask,
        target_frame_instance_mask=target_frame_instance_mask,
    )

    prompt = "police officer"
    reactor_object = ReactorObject(
        new_class_id=200,
        prompts=[prompt],
        registry_class_name="Pedestrian",
        change_shape=False,
        replacement_probability=replacement_probability,
    )
    reactor_object.registry_class_id = 22
    instance_mask, id_to_prompt = get_instance_mask_and_prompts(
        camera_reactor_input=camera_reactor_input, reactor_object=reactor_object
    )
    if replacement_probability == 1.0:
        assert len(id_to_prompt.keys()) == 3
        assert id_to_prompt[1] == prompt
        assert id_to_prompt[2] == prompt
        assert id_to_prompt[3] == prompt
        assert np.array_equal(np.unique(instance_mask), np.array([0, 1, 2, 3]))
        assert instance_mask[0, 0, 0] == 1
        assert instance_mask[4, 4, 0] == 3
        assert instance_mask[12, 12, 0] == 2
    if replacement_probability == 0.0:
        assert len(id_to_prompt.keys()) == 0
        assert len(np.unique(instance_mask)) == 1


@pytest.mark.parametrize(
    "asset_name",
    ["char_alvin_004", "truck_shipping_02"],
)
def test_get_instance_mask_and_prompts_from_paired_frame(
    asset_name: str,
    target_image: np.ndarray,
    target_depth: np.ndarray,
    target_frame_semseg_mask: SemanticSegmentation2D,
    target_frame_instance_mask: InstanceSegmentation2D,
    empty_image: np.ndarray,
    empty_frame_semseg_mask: SemanticSegmentation2D,
    empty_frame_instance_mask: InstanceSegmentation2D,
):
    camera_reactor_input = PerCameraReactorInput(
        target_image=target_image,
        target_depth=target_depth,
        target_frame_semseg_mask=target_frame_semseg_mask,
        target_frame_instance_mask=target_frame_instance_mask,
        empty_image=empty_image,
        empty_frame_semseg_mask=empty_frame_semseg_mask,
        empty_frame_instance_mask=empty_frame_instance_mask,
    )
    prompt = "police officer"
    reactor_object = ReactorObject(
        new_class_id=200,
        prompts=[prompt],
        asset_name=asset_name,
        change_shape=True,
    )
    if asset_name == "char_alvin_004":
        reactor_object.registry_class_id = 22
    else:
        reactor_object.registry_class_id = 2

    instance_mask, id_to_prompt = get_instance_mask_and_prompts(
        camera_reactor_input=camera_reactor_input, reactor_object=reactor_object
    )
    if asset_name == "char_alvin_004":
        assert len(id_to_prompt.keys()) == 1
        assert id_to_prompt[1] == prompt
        assert np.array_equal(np.unique(instance_mask), np.array([0, 1]))
        assert instance_mask[12, 12, 0] == 1
    if asset_name == "truck_shipping_02":
        # No match in semseg mask with class_id of this asset
        assert id_to_prompt is None
        assert np.array_equal(np.unique(instance_mask), np.array([0]))


@pytest.mark.parametrize(
    "color_channels",
    [1, 3, 4],
)
def test_encode_decode_rgba_image_base64(color_channels: int):
    img = np.random.randint(low=0, high=254, size=(108, 192, color_channels), dtype=np.uint8)
    img_str = encode_rgba_image_png_base64(image=img)
    with TemporaryDirectory() as tmp_dir:
        json_path = os.path.join(tmp_dir, "rgba.json")
        with open(json_path, "w") as outfile:
            json.dump({"img": img_str}, outfile, ensure_ascii=False, indent=4)

        response = json.loads(open(json_path).read())
        img_decode = decode_rgba_image_png_base64(image_str=response["img"])
        if color_channels == 1:
            img_decode = np.expand_dims(img_decode, 2)
        assert np.array_equal(img, img_decode)
