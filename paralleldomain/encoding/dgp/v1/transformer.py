import logging

import numpy as np

from paralleldomain.encoding.encoder import MaskTransformer, ObjectTransformer
from paralleldomain.utilities.mask import encode_int32_as_rgb8

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


class SemanticSegmentation3DTransformer(MaskTransformer):
    @staticmethod
    def _transform(mask: np.ndarray) -> np.ndarray:
        return mask.astype(np.uint32)


class InstanceSegmentation3DTransformer(SemanticSegmentation3DTransformer):
    ...


class KeyPoint2DTransformer(ObjectTransformer):
    ...


class KeyLine2DTransformer(ObjectTransformer):
    ...


class Polygon2DTransformer(ObjectTransformer):
    ...


class KeyPoint3DTransformer(ObjectTransformer):
    ...


class KeyLine3DTransformer(ObjectTransformer):
    ...


class Polygon3DTransformer(ObjectTransformer):
    ...
