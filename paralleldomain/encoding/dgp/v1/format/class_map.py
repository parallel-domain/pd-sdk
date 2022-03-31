from typing import Tuple, Union

from paralleldomain.common.dgp.v1 import annotations_pb2
from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP_INV, DirectoryName
from paralleldomain.encoding.dgp.v1.format.common import CLASS_MAPS_KEY, CUSTOM_FORMAT_KEY, CommonDGPV1FormatMixin
from paralleldomain.encoding.dgp.v1.utils import _attribute_key_dump, _attribute_value_dump
from paralleldomain.encoding.pipeline_encoder import PipelineItem
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class ClassMapDGPV1Mixin(CommonDGPV1FormatMixin):
    def save_class_map_in_state(self, pipeline_item: PipelineItem, data: Tuple[ClassMap, AnnotationType]):
        class_map, annotype = data
        pipeline_item.custom_data[CUSTOM_FORMAT_KEY][CLASS_MAPS_KEY][str(ANNOTATION_TYPE_MAP_INV[annotype])] = class_map
