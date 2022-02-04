import logging
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Union

from paralleldomain.encoding.dgp.v1.encoder_steps.pipeline_builder import DGPV1PipelineBuilder
from paralleldomain.model.type_aliases import FrameId

try:
    import pypeln
except ImportError:
    pypeln = None


from paralleldomain import Scene
from paralleldomain.encoding.pipeline_encoder import DatasetPipelineEncoder, EncoderStep, FinalStep
from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger(__name__)


class DGPV1DatasetPipelineEncoder(DatasetPipelineEncoder[Scene, Dict[str, Any]]):
    @classmethod
    def from_path(
        cls,
        dataset_path: Union[str, AnyPath],
        output_path: Union[str, AnyPath],
        dataset_format: str,
        encoder_steps_builder: Optional[Callable[[], List[EncoderStep]]] = None,
        final_encoder_step_builder: Optional[Callable[[], FinalStep]] = None,
        sensor_names: Optional[Union[List[str], Dict[str, str]]] = None,
        sim_offset: float = 0.01 * 5,
        allowed_frames: Optional[List[FrameId]] = None,
        stages_max_out_queue_size: int = 3,
        workers_per_step: int = 2,
        max_queue_size_per_step: int = 4,
        scene_names: Optional[List[str]] = None,
        set_start: Optional[int] = None,
        set_stop: Optional[int] = None,
        use_tqdm: bool = True,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "DatasetPipelineEncoder":
        pipeline_builder = DGPV1PipelineBuilder(
            output_path=output_path,
            sim_offset=sim_offset,
            sensor_names=sensor_names,
            encoder_steps_builder=encoder_steps_builder,
            final_encoder_step_builder=final_encoder_step_builder,
            stages_max_out_queue_size=stages_max_out_queue_size,
            allowed_frames=allowed_frames,
            workers_per_step=workers_per_step,
            max_queue_size_per_step=max_queue_size_per_step,
        )
        return DatasetPipelineEncoder.from_path_and_builder(
            dataset_path=dataset_path,
            use_tqdm=use_tqdm,
            dataset_format=dataset_format,
            set_stop=set_stop,
            set_start=set_start,
            scene_names=scene_names,
            decoder_kwargs=decoder_kwargs,
            pipeline_builder=pipeline_builder,
        )
