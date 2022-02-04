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
    def apply(self, scene: Scene, input_stage: Iterable[Any]) -> Iterable[Any]:
        pass


class FinalStep(Generic[T]):
    @abstractmethod
    def aggregate(self, scene: Scene, input_stage: Iterable[Any]) -> Iterable[Any]:
        pass

    @abstractmethod
    def finalize(self) -> T:
        pass


class PipelineBuilder(Generic[S, T]):
    @abstractmethod
    def build_scene_aggregator(self, dataset: Dataset) -> SceneAggregator[T]:
        pass

    @abstractmethod
    def build_scene_encoder_steps(self, dataset: Dataset, scene: S) -> List[EncoderStep]:
        pass

    @abstractmethod
    def build_scene_final_encoder_step(self, dataset: Dataset, scene: S) -> FinalStep[T]:
        pass

    @abstractmethod
    def build_pipeline_source_generator(self, dataset: Dataset, scene: S) -> Generator[Dict[str, Any], None, None]:
        pass


class DatasetPipelineEncoder(Generic[S, T]):
    def __init__(
        self,
        dataset: Dataset,
        pipeline_builder: PipelineBuilder,
        scene_names: Optional[List[str]] = None,
        set_start: Optional[int] = None,
        set_stop: Optional[int] = None,
        use_tqdm: bool = True,
    ):
        self.pipeline_builder = pipeline_builder
        self.use_tqdm = use_tqdm
        self._dataset = dataset

        if scene_names is not None:
            for sn in scene_names:
                if sn not in self._dataset.unordered_scene_names:
                    raise KeyError(f"{sn} could not be found in dataset {self._dataset.name}")
            self._scene_names = scene_names
        else:
            set_slice = slice(set_start, set_stop)
            self._scene_names = self._dataset.unordered_scene_names[set_slice]

    def encode_dataset(self):
        scene_aggregator = self.pipeline_builder.build_scene_aggregator(dataset=self._dataset)
        for scene_name in self._scene_names:
            scene = self._dataset.get_unordered_scene(scene_name=scene_name)
            scene_info = self._encode_scene(scene=scene)

            scene_aggregator.aggregate(scene_info=scene_info)
        scene_aggregator.finalize()

    def _encode_scene(self, scene: S):
        stage = self.pipeline_builder.build_pipeline_source_generator(dataset=self._dataset, scene=scene)
        encoder_steps = self.pipeline_builder.build_scene_encoder_steps(dataset=self._dataset, scene=scene)
        final_encoder_step = self.pipeline_builder.build_scene_final_encoder_step(dataset=self._dataset, scene=scene)
        for encoder in encoder_steps:
            stage = encoder.apply(scene=scene, input_stage=stage)

        stage = final_encoder_step.aggregate(scene=scene, input_stage=stage)

        if self.use_tqdm:
            stage = tqdm(stage)

        pypeln.process.run(stage)

        return final_encoder_step.finalize()

    @classmethod
    def from_path_and_builder(
        cls,
        dataset_path: Union[str, AnyPath],
        dataset_format: str,
        pipeline_builder: PipelineBuilder,
        scene_names: Optional[List[str]] = None,
        set_start: Optional[int] = None,
        set_stop: Optional[int] = None,
        use_tqdm: bool = True,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "DatasetPipelineEncoder":
        if decoder_kwargs is None:
            decoder_kwargs = dict()
        dataset = decode_dataset(dataset_path=dataset_path, dataset_format=dataset_format, **decoder_kwargs)
        return cls(
            dataset=dataset,
            scene_names=scene_names,
            set_start=set_start,
            set_stop=set_stop,
            use_tqdm=use_tqdm,
            pipeline_builder=pipeline_builder,
        )
