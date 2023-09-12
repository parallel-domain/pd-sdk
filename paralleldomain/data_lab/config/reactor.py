from dataclasses import dataclass, field
from typing import List

from pd.internal.assets.asset_registry import InfoSegmentation


def get_asset_registry_class_id(asset_name: str) -> int:
    segmentation_info = InfoSegmentation.select().where((InfoSegmentation.name % asset_name))
    class_ids = [o.panoptic_id.pk_id for o in segmentation_info]
    if len(class_ids) == 0:
        raise ValueError(f"Did not find class_id in asset registry for {asset_name}.")
    elif len(class_ids) > 1:
        raise ValueError(f"Found multiple class_ids in asset registry for {asset_name}.")
    else:
        return class_ids[0]


@dataclass
class ReactorObject:
    """
    Args:
        new_class_id: int
        prompts: List[str]
        asset_name: str
        registry_class_id: int
    Attributes:
        new_class_id: class_id to assign to generated object
        prompts: prompts are guiding the Reactor process towards which objects to generate
        asset_name: target asset name that provides object mask for Reactor
        registry_class_id: class_id stored in the asset registry, will be retrieved automatically
    """

    new_class_id: int
    prompts: List[str]
    asset_name: str
    registry_class_id: int = None

    def init_asset_registry_class_id(self):
        if self.registry_class_id is None:
            self.registry_class_id = get_asset_registry_class_id(asset_name=self.asset_name)


@dataclass
class ReactorConfig:
    """Parameters for Reactor.
    Args:
        reactor_object: List
        number_of_inference_steps: int
        guidance_scale: float
        noise_scheduler: str
        use_color_matching: bool
        use_camera_noise: bool
        clip_score_threshold: float
        cameras_to_use: List
        random_seed: int
        negative_prompt: str
        inference_resolution: int
        context_scale: float
        color_matching_strength: float
        undistort_input: bool
        undistort_context_scale_pad_factor: float
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
