from datetime import datetime
from functools import partial
from typing import Any, Dict, Generator, Iterable, List, Optional

import pypeln

from paralleldomain import Scene
from paralleldomain.common.dgp.v1 import annotations_pb2
from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP_INV, DirectoryName
from paralleldomain.encoding.dgp.v1.encoder_steps.helper import EncoderStepHelper
from paralleldomain.encoding.dgp.v1.utils import _attribute_key_dump, _attribute_value_dump
from paralleldomain.encoding.pipeline_encoder import EncoderStep
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes, BoundingBox2D
from paralleldomain.model.sensor import CameraSensorFrame, FilePathedDataType
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.mask import encode_2int16_as_rgba8, encode_int32_as_rgb8


class SceneFlowEncoderStep(EncoderStepHelper, EncoderStep):
    def __init__(
        self,
        fs_copy: bool,
        workers: int = 1,
        in_queue_size: int = 1,
        output_annotation_types: Optional[List[AnnotationType]] = None,
    ):
        self.fs_copy = fs_copy
        self.in_queue_size = in_queue_size
        self.workers = workers
        self.output_annotation_types = output_annotation_types

    @property
    def apply_stages(self) -> bool:
        return self.output_annotation_types is None or AnnotationTypes.SceneFlow in self.output_annotation_types

    def encode_scene_flow(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        sensor_frame = self._get_lidar_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is not None and AnnotationTypes.SceneFlow in sensor_frame.available_annotation_types:
            output_path = self._get_dgpv1_file_output_path(
                sensor_frame=sensor_frame,
                input_dict=input_dict,
                file_suffix="npz",
                directory_name=DirectoryName.MOTION_VECTORS_3D,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf_file_path = sensor_frame.get_file_path(data_type=FilePathedDataType.SceneFlow)
            if self.fs_copy and sf_file_path is not None:
                save_path = fsio.copy_file(source=sf_file_path, target=output_path)
            else:
                scene_flow = sensor_frame.get_annotations(AnnotationTypes.SceneFlow)
                save_path = fsio.write_npz(obj=dict(motion_vectors=scene_flow.vectors), path=output_path)

            if "annotations" not in input_dict:
                input_dict["annotations"] = dict()
            input_dict["annotations"][str(ANNOTATION_TYPE_MAP_INV[AnnotationTypes.SceneFlow])] = save_path
        return input_dict

    def apply(self, input_stage: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        stage = input_stage
        if self.apply_stages:
            stage = pypeln.thread.map(
                f=self.encode_scene_flow, stage=input_stage, workers=self.workers, maxsize=self.in_queue_size
            )

        return stage
