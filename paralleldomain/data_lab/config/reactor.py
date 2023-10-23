from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.projection import DistortionLookupTable

# assumes we use dgp ontology
PD_ASSET_DETAILS_LE = {
    "asset_semantic_map": {
        "SM_primitive_box_1m": 200,
        "SM_primitive_torus_1m": 200,
        "SM_primitive_stroller_01": 200,
    }
}


def extract_distortion_lookup_tables(distortion_lookup_files: Dict[str, str]) -> Dict[str, DistortionLookupTable]:
    distortion_lookup_tables = dict()
    for camera_name, distortion_lookup_file in distortion_lookup_files.items():
        distortion_lookup_file = AnyPath(distortion_lookup_file)
        if distortion_lookup_file.exists():
            with distortion_lookup_file.open() as f:
                lut = np.loadtxt(f, delimiter=",", dtype="float")
            lookup = DistortionLookupTable.from_ndarray(lut)
            distortion_lookup_tables[camera_name] = lookup
        else:
            raise ValueError(f"Can't find distortion lookup table file {distortion_lookup_file}")
    return distortion_lookup_tables


@dataclass
class ReactorObject:
    """
    Attributes:
        prompts: prompts are guiding the Reactor process towards which objects to generate
        new_class_id: class_id to assign to generated object
        asset_name: target asset name that provides object mask for Reactor
        change_shape:
            True: mode to convert primitive shapes (e.g. a cube) into a new object with different texture and shape.
            False: mode to texturize an existing object, while maintaining the same shape.
        replacement_probability: When set to `1.0` all object instances will be replaced.
            only used when `change_shape = False`
        registry_class_name: object instances of this class name will be texturized by Reactor.
        registry_class_id: class_id stored in the asset registry, will be retrieved automatically
    """

    prompts: List[str]
    new_class_id: int = None
    asset_name: Optional[str] = None
    change_shape: bool = True
    replacement_probability: float = 1.0
    registry_class_name: Optional[str] = None
    registry_class_id: int = None

    def set_asset_registry_class_id(self, class_map: Optional[ClassMap] = None):
        if self.asset_name is not None and self.registry_class_name is None:
            supported_assets = list(PD_ASSET_DETAILS_LE["asset_semantic_map"].keys())
            if self.asset_name in supported_assets:
                self.registry_class_id = PD_ASSET_DETAILS_LE["asset_semantic_map"][self.asset_name]
            else:
                raise ValueError(
                    f"Reactor does not support the spawned asset {self.asset_name}. "
                    f"Use one of the following assets {supported_assets} instead."
                )
        elif class_map is not None and self.registry_class_name is not None:
            class_detail = class_map.get_class_detail_from_name(class_name=self.registry_class_name)
            if class_detail is not None:
                self.registry_class_id = class_detail.id
            else:
                raise ValueError(
                    f"Did not find class name {self.registry_class_name} in ontology {class_map.class_names}."
                )
        else:
            raise ValueError(
                "Need to provide either asset name or class_map and registry_class_name to set registry_class_id."
            )

    def __post_init__(self):
        if self.change_shape is True:
            if self.replacement_probability < 1.0:
                raise NotImplementedError("Parameter replacement_probability not supported if change_shape is used.")
            if self.asset_name is None:
                raise NotImplementedError("Parameter asset_name is required if change_shape is used.")
            if self.new_class_id is None:
                self.new_class_id = 200


@dataclass
class ReactorConfig:
    """Parameters for Reactor.

    Attributes:
        reactor_object: defines the object to be modified using a prompt
        number_of_inference_steps: diffusion steps of the image generation model
        guidance_scale: controls how much the image generation process follows the text prompt
        noise_scheduler: noise scheduler of the image generation model
        use_color_matching: applies hue color transform on the generated object
        use_camera_noise: applies camera noise on the generated object
        clip_score_threshold: rejects generated samples with a clip score below threshold. Good range between 18-26.
        cameras_to_use: which cameras to use for Reactor
        random_seed: is passed to the Reactor model
        negative_prompt: helps guide the Reactor diffusion process.
        inference_resolution: image crop will be scaled to this resolution before the image generation process
        context_scale: amount of context to add around the object instance mask.
        context_scale = 1 no additional context.
        context_scale 2 = add padding such that resulting crop is twice as large as the original mask.
        color_matching_strength: strength of color matching. 0.0 = no strength, 1.0 = full strength.
        undistort_input: whether to undistort fisheye inputs
        undistort_context_scale_pad_factor: how much padding to add before undistorting
        noise_strength: strength of noise applied to the image generation model, only used when change_shape = False
        distortion_lookups: maps cameras_to_use to DistortionLookupTable.
            Required for sensor rigs with pd_fisheye cameras. Helper function: extract_distortion_lookup_tables
    """

    reactor_object: ReactorObject
    number_of_inference_steps: int = 20
    guidance_scale: float = 7.5
    noise_scheduler: str = "dpm++"
    use_color_matching: bool = False
    use_camera_noise: bool = False
    clip_score_threshold: float = 18
    cameras_to_use: List = field(default_factory=lambda: ["Front"])
    random_seed: int = 42
    negative_prompt: str = "comic style, drawing, motion blur, gray cube, white sign, warped, artifacts"
    inference_resolution: int = 512
    context_scale: float = 2.0
    color_matching_strength: float = 0.5
    undistort_input: bool = True
    undistort_context_scale_pad_factor: float = 1.5
    noise_strength: Optional[float] = 0.5
    distortion_lookups: Optional[Dict[str, DistortionLookupTable]] = None
