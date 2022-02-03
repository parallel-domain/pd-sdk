import logging
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Any, Dict, Generator, List, Optional, Union

from tqdm import tqdm

from paralleldomain.encoding.dgp.v1.encoder_steps.encoder_step import EncoderStep
from paralleldomain.encoding.dgp.v1.encoder_steps.encoder_steps import EncoderSteps
from paralleldomain.model.sensor import CameraSensor, LidarSensor

try:
    import pypeln
except ImportError:
    pypeln = None


from paralleldomain import Dataset, Scene
from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.encoding.pipeline_encoder import DatasetPipelineEncoder, S
from paralleldomain.utilities.any_path import AnyPath

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


class DGPV1DatasetPipelineEncoder(DatasetPipelineEncoder[Scene]):
    def __init__(
        self,
        dataset: Dataset,
        dataset_path: AnyPath,
        output_path: AnyPath,
        encoder_steps: List[EncoderStep],
        sensor_names: Optional[Union[List[str], Dict[str, str]]] = None,
        scene_names: Optional[List[str]] = None,
        set_start: Optional[int] = None,
        set_stop: Optional[int] = None,
        sim_offset: float = 0.01 * 5,
        use_tqdm: bool = True,
        **decoder_kwargs,
    ):
        super().__init__(dataset=dataset, scene_names=scene_names, set_stop=set_stop, set_start=set_start)
        self.use_tqdm = use_tqdm
        self.encoder_steps = encoder_steps
        self.sensor_names = sensor_names
        self.dataset_format = dataset.format
        self.sim_offset = sim_offset
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.decoder_kwargs = decoder_kwargs

    def _encode_scene(self, scene: S, source_generator: Generator[Dict[str, Any], None, None]):

        stage = source_generator
        for encoder in self.encoder_steps:
            stage = encoder.apply(scene=scene, input_stage=stage)

        aggregated_results = list()
        result_pipeline = pypeln.sync.to_iterable(stage, maxsize=5, return_index=False)
        if self.use_tqdm:
            result_pipeline = tqdm(result_pipeline)
        for item in result_pipeline:
            aggregated_results.append(item)

        pass

    def _pipeline_source_generator(self, scene: Scene) -> Generator[Dict[str, Any], None, None]:
        if self.sensor_names is None:
            sensor_name_mapping = {s: s for s in scene.sensor_names}
        elif isinstance(self.sensor_names, list):
            sensor_name_mapping = {s: s for s in self.sensor_names if s in scene.sensor_names}
        elif isinstance(self.sensor_names, dict):
            sensor_name_mapping = {t: s for t, s in self.sensor_names.items() if s in scene.sensor_names}
        else:
            raise ValueError(f"sensor_names is neither a list nor a dict but {type(self.sensor_names)}!")

        reference_timestamp: datetime = scene.get_frame(scene.frame_ids[0]).date_time
        output_path = self.output_path / scene.name

        logger.info(f"Encoding Scene {scene.name} with sensor mapping: {sensor_name_mapping}")
        for target_sensor_name, source_sensor_name in sensor_name_mapping.items():
            sensor = scene.get_sensor(sensor_name=source_sensor_name)
            if sensor.name in sensor_name_mapping:
                for sensor_frame in sensor.sensor_frames:
                    if isinstance(sensor, CameraSensor):
                        yield dict(
                            camera_frame_info=dict(
                                sensor_name=sensor.name,
                                frame_id=sensor_frame.frame_id,
                                scene_name=scene.name,
                                dataset_path=self.dataset_path,
                                dataset_format=self.dataset_format,
                                decoder_kwargs=self.decoder_kwargs,
                            ),
                            target_sensor_name=target_sensor_name,
                            scene_output_path=output_path,
                            scene_reference_timestamp=reference_timestamp,
                            sim_offset=self.sim_offset,
                        )
                    elif isinstance(sensor, LidarSensor):
                        yield dict(
                            lidar_frame_info=dict(
                                sensor_name=sensor.name,
                                frame_id=sensor_frame.frame_id,
                                scene_name=scene.name,
                                dataset_path=self.dataset_path,
                                dataset_format=self.dataset_format,
                                decoder_kwargs=self.decoder_kwargs,
                            ),
                            target_sensor_name=target_sensor_name,
                            scene_output_path=output_path,
                            scene_reference_timestamp=reference_timestamp,
                            sim_offset=self.sim_offset,
                        )

    @classmethod
    def from_path(
        cls,
        dataset_path: AnyPath,
        dataset_format: str,
        output_path: AnyPath,
        encoder_steps: List[EncoderStep],
        sensor_names: Optional[Union[List[str], Dict[str, str]]] = None,
        scene_names: Optional[List[str]] = None,
        set_start: Optional[int] = None,
        set_stop: Optional[int] = None,
        sim_offset: float = 0.01 * 5,
        **decoder_kwargs,
    ) -> "DGPV1DatasetPipelineEncoder":
        dataset = decode_dataset(dataset_path=dataset_path, dataset_format=dataset_format, **decoder_kwargs)
        return cls(
            dataset=dataset,
            output_path=output_path,
            scene_names=scene_names,
            set_start=set_start,
            sim_offset=sim_offset,
            set_stop=set_stop,
            dataset_path=dataset_path,
            sensor_names=sensor_names,
            encoder_steps=encoder_steps,
        )


if __name__ == "__main__":
    with TemporaryDirectory() as temp_dir:
        encoder = DGPV1DatasetPipelineEncoder.from_path(
            dataset_path=AnyPath("/home/phillip/data/d00e377f-f69d-48a3-a208-620a00fecfd3"),
            dataset_format="dgpv1",
            output_path=AnyPath(temp_dir),
            set_start=0,
            set_stop=1,
            sensor_names={"FCM_front": "FCM_front", "FCM_front2": "FCM_front"},
            encoder_steps=EncoderSteps.get_default(workers_per_step=1, max_queue_size_per_step=4),
        )
        encoder.encode_dataset()
        pass
