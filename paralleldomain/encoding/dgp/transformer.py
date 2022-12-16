import logging

import numpy as np

from paralleldomain.encoding.encoder import MaskTransformer, ObjectTransformer
from paralleldomain.utilities.mask import encode_2int16_as_rgba8, encode_int32_as_rgb8

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
        # Constrain float between 0.0 and 1.0 then convert to uint16
        height, width = mask.shape[:2]
        mask /= 2 * np.array([width, height])
        mask += 0.5
        mask_2int16 = (mask * 65535).astype(np.uint16)
        return encode_2int16_as_rgba8(mask_2int16)


class SemanticSegmentation3DTransformer(MaskTransformer):
    @staticmethod
    def _transform(mask: np.ndarray) -> np.ndarray:
        return mask.astype(np.uint32)


class InstanceSegmentation3DTransformer(SemanticSegmentation3DTransformer):
    ...
