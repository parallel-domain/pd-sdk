import os
from tempfile import TemporaryDirectory
from typing import List

import numpy as np

from paralleldomain import Dataset, Scene
from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.encoding.data_stream.encoding_format import DataStreamEncodingFormat
from paralleldomain.encoding.helper import encode_dataset
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.model.type_aliases import FrameId
from paralleldomain.utilities.dataset_transform import DatasetTransformation, DataTransformer


def test_encode_decode(dataset: Dataset):
    if "SKIP_ENCODER" not in os.environ:
        with TemporaryDirectory() as temp_dir:
            encoding_format = DataStreamEncodingFormat(
                output_path=temp_dir,
            )
            target_scenes = dataset.scene_names[:2]

            class FilterFrames(DataTransformer[List[FrameId], Scene]):
                def __call__(self, model: Scene) -> List[FrameId]:
                    return model.frame_ids[:3]

            data_transform = DatasetTransformation(
                frame_ids=FilterFrames(),
                scene_names=target_scenes,
            )
            dataset = data_transform.apply(dataset)

            encode_dataset(
                dataset=dataset,
                encoding_format=encoding_format,
                copy_all_available_sensors_and_annotations=True,
                workers=4,
                max_in_queue_size=12,
                run_env="thread",
            )

            dataset_decoded = decode_dataset(dataset_path=temp_dir, dataset_format="data-stream")
            assert len(dataset_decoded.scene_names) == len(target_scenes)
            for sf, frame, scene in dataset_decoded.sensor_frame_pipeline():
                frame_id_idx = scene.frame_ids.index(frame.frame_id)

                target_scene = dataset.get_scene(scene_name=scene.name)
                target_frame_id = target_scene.frame_ids[frame_id_idx]

                target_frame = target_scene.get_frame(frame_id=target_frame_id)
                sensor_name = next(iter([sn for sn in target_frame.sensor_names if sf.sensor_name.startswith(sn)]))
                og_sf = target_frame.get_sensor(sensor_name=sensor_name)
                if isinstance(sf, CameraSensorFrame):
                    assert isinstance(og_sf, CameraSensorFrame)
                    assert np.allclose(sf.image.rgba, og_sf.image.rgba)
