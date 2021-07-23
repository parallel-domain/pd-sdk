import json
import logging
from typing import Dict, List, Union

import numpy as np
from PIL import Image

from paralleldomain.encoding.utils.log import setup_loggers
from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger("fsio")
setup_loggers(["fsio"], log_level=logging.DEBUG)


def write_json(obj: Union[Dict, List], path: AnyPath):
    with path.open("w") as fp:
        json.dump(obj, fp, indent=2)
    logger.debug(f"Finished writing {str(path)}")


def write_png(obj: np.ndarray, path: AnyPath):
    with path.open("wb") as fp:
        Image.fromarray(obj).save(fp, format="png")
    logger.debug(f"Finished writing {str(path)}")


def write_npz(obj: Dict[str, np.ndarray], path: AnyPath):
    with path.open("wb") as fp:
        np.savez(fp, **obj)
    logger.debug(f"Finished writing {str(path)}")
