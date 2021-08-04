import logging

import numpy as np

from paralleldomain.encoding.encoder import MaskTransformer, ObjectTransformer
from paralleldomain.encoding.utilities.mask import encode_2int16_as_rgba8, encode_int32_as_rgb8

logger = logging.getLogger(__name__)


class BoundingBox2DTransformer(ObjectTransformer):
    ...


class BoundingBox3DTransformer(ObjectTransformer):
    ...


class SemanticSegmentation2DTransformer(MaskTransformer):
    @staticmethod
    def _transform(mask: np.ndarray) -> np.ndarray:
        return encode_int32_as_rgb8(mask)


class InstanceSegmentation2DTransformer(SemanticSegmentation2DTransformer):
    ...


class OpticalFlowTransformer(MaskTransformer):
    @staticmethod
    def _transform(mask: np.ndarray) -> np.ndarray:
        return encode_2int16_as_rgba8(mask)


class SemanticSegmentation3DTransformer(MaskTransformer):
    @staticmethod
    def _transform(mask: np.ndarray) -> np.ndarray:
        return mask.astype(np.uint32)


class InstanceSegmentation3DTransformer(SemanticSegmentation3DTransformer):
    ...
