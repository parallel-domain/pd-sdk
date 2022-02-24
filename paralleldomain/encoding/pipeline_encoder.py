import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, Generator, Generic, Iterable, List, Optional, Tuple, TypeVar, Union

from tqdm import tqdm

try:
    import pypeln
except ImportError:
    pypeln = None


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


class EncoderStep:
    @abstractmethod
    def apply(self, input_stage: Iterable[Any]) -> Iterable[Any]:
        pass


class PipelineBuilder:
    @abstractmethod
    def build_encoder_steps(self) -> List[EncoderStep]:
        pass

    @abstractmethod
    def build_pipeline_source_generator(self) -> Generator[Dict[str, Any], None, None]:
        pass

    @property
    @abstractmethod
    def pipeline_item_unit_name(self):
        pass


class DatasetPipelineEncoder:
    def __init__(
        self,
        pipeline_builder: PipelineBuilder,
        use_tqdm: bool = True,
    ):
        self.pipeline_builder = pipeline_builder
        self.use_tqdm = use_tqdm

    def encode_dataset(self):
        stage = self.pipeline_builder.build_pipeline_source_generator()
        encoder_steps = self.pipeline_builder.build_encoder_steps()
        for encoder in encoder_steps:
            stage = encoder.apply(input_stage=stage)

        stage = pypeln.process.to_iterable(stage)
        if self.use_tqdm:
            stage = tqdm(
                stage,
                desc="Encoding Progress",
                unit=f" {self.pipeline_builder.pipeline_item_unit_name}",
                smoothing=0.0,
            )
        for _ in stage:
            pass

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
