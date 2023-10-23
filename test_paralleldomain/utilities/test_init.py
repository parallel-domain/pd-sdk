import logging
import unittest

import numpy as np

from paralleldomain.utilities import clip_with_warning


class ListHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def emit(self, record):
        self.records.append(record)


class TestClipWithWarning(unittest.TestCase):
    def test_npuint16_clip(self):
        logger = logging.getLogger()
        handler = ListHandler()
        logger.addHandler(handler)

        arr1 = np.array([50000, 70000, 100000, 5000, -10])
        clipped_arr1 = clip_with_warning(arr1, np.uint16)
        assert np.array_equal(clipped_arr1, np.array([50000, 65535, 65535, 5000, 0]))
        assert any(record.levelno == logging.WARNING for record in handler.records)

    def test_npint8_clip(self):
        logger = logging.getLogger()
        handler = ListHandler()
        logger.addHandler(handler)

        arr2 = np.array([-10, 128, 256, 50])
        clipped_arr2 = clip_with_warning(arr2, np.int8)
        assert np.array_equal(clipped_arr2, np.array([-10, 127, 127, 50]))
        assert any(record.levelno == logging.WARNING for record in handler.records)

    def test_npint8_noclip(self):
        logger = logging.getLogger()
        handler = ListHandler()
        logger.addHandler(handler)

        arr3 = np.array([0, 2, 3, 127])
        clipped_arr3 = clip_with_warning(arr3, np.int8)  # no clipping should occurr
        assert np.array_equal(clipped_arr3, arr3)
        assert not (any(record.levelno == logging.WARNING for record in handler.records))


if __name__ == "__main__":
    unittest.main()
