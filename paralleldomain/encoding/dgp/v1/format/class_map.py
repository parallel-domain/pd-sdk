from typing import Tuple

from paralleldomain.encoding.dgp.v1.format.common import CLASS_MAPS_KEY, CUSTOM_FORMAT_KEY, CommonDGPV1FormatMixin
from paralleldomain.encoding.pipeline_encoder import PipelineItem
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.class_mapping import ClassMap


class ClassMapDGPV1Mixin(CommonDGPV1FormatMixin):
    def save_class_map_in_state(self, pipeline_item: PipelineItem, data: Tuple[ClassMap, AnnotationType]):
        class_map, annotype = data
        pipeline_item.custom_data[CUSTOM_FORMAT_KEY][CLASS_MAPS_KEY][
            str(self._annotation_type_map_inv[annotype])
        ] = class_map
