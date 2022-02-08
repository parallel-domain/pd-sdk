import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, Generator, Generic, Iterable, List, Optional, Tuple, TypeVar, Union

from tqdm import tqdm

from paralleldomain import Dataset, Scene
from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.model.type_aliases import SceneName
from paralleldomain.model.unordered_scene import UnorderedScene
from paralleldomain.utilities.any_path import AnyPath

try:
    import pypeln
except ImportError:
    pypeln = None


S = TypeVar("S", UnorderedScene, Scene)
T = TypeVar("T")

logger = logging.getLogger(__name__)


class _TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


logger.addHandler(_TqdmLoggingHandler())

ProcessDataStage = Iterable[Any]
SaveDataStage = Iterable[Any]
StepResultStages = Tuple[ProcessDataStage, SaveDataStage]


class SceneAggregator(Generic[T]):
    @abstractmethod
    def aggregate(self, scene_info: T):
        pass

    @abstractmethod
    def finalize(self):
        pass


class EncoderStep:
    @abstractmethod
    def apply(self, input_stage: Iterable[Any]) -> Iterable[Any]:
        pass


class FinalStep(Generic[T]):
    @abstractmethod
    def aggregate(self, input_stage: Iterable[Any]) -> Iterable[Any]:
        pass

    @abstractmethod
    def finalize(self) -> T:
        pass


class PipelineBuilder(Generic[S, T]):
    @abstractmethod
    def build_scene_aggregator(self) -> SceneAggregator[T]:
        pass

    @abstractmethod
    def build_scene_encoder_steps(self, dataset: Dataset, scene: S, **kwargs) -> List[EncoderStep]:
        pass

    @abstractmethod
    def build_scene_final_encoder_step(self, dataset: Dataset, scene: S, **kwargs) -> FinalStep[T]:
        pass

    @abstractmethod
    def build_pipeline_source_generator(
        self, dataset: Dataset, scene: S, **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        pass

    @abstractmethod
    def build_pipeline_scene_generator(self) -> Generator[Tuple[Dataset, S, Dict[str, Any]], None, None]:
        pass

    @property
    @abstractmethod
    def pipeline_item_unit_name(self):
        pass


class DatasetPipelineEncoder(Generic[S, T]):
    def __init__(
        self,
        pipeline_builder: PipelineBuilder,
        use_tqdm: bool = True,
    ):
        self.pipeline_builder = pipeline_builder
        self.use_tqdm = use_tqdm

    def encode_dataset(self):
        scene_aggregator = self.pipeline_builder.build_scene_aggregator()
        for dataset, scene, kwargs in self.pipeline_builder.build_pipeline_scene_generator():
            scene_info = self._encode_scene(scene=scene, dataset=dataset, **kwargs)
            scene_aggregator.aggregate(scene_info=scene_info)
        scene_aggregator.finalize()

    def _encode_scene(self, dataset: Dataset, scene: S, **kwargs):
        stage = self.pipeline_builder.build_pipeline_source_generator(dataset=dataset, scene=scene, **kwargs)
        encoder_steps = self.pipeline_builder.build_scene_encoder_steps(dataset=dataset, scene=scene, **kwargs)
        final_encoder_step = self.pipeline_builder.build_scene_final_encoder_step(
            dataset=dataset, scene=scene, **kwargs
        )
        for encoder in encoder_steps:
            stage = encoder.apply(input_stage=stage)

        stage = final_encoder_step.aggregate(input_stage=stage)

        if self.use_tqdm:
            stage = tqdm(
                stage, desc=f"{scene.name}", unit=f" {self.pipeline_builder.pipeline_item_unit_name}", smoothing=0.0
            )

        pypeln.process.run(stage)

        return final_encoder_step.finalize()

    @classmethod
    def from_builder(
        cls,
        pipeline_builder: PipelineBuilder,
        use_tqdm: bool = True,
        **kwargs,
    ) -> "DatasetPipelineEncoder":
        return cls(
            use_tqdm=use_tqdm,
            pipeline_builder=pipeline_builder,
        )
