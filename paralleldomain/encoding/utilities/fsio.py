import hashlib
import json
import logging
import os
from typing import Dict, List, Union

import numpy as np
from PIL import Image

from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger("fsio")


def write_json(obj: Union[Dict, List], path: AnyPath, append_sha1: bool = False):
    json_obj = json.dumps(obj, indent=2)

    if append_sha1:
        # noinspection InsecureHash
        json_obj_sha1 = hashlib.sha1(json_obj.encode()).hexdigest()
        filename_sha1 = (
            f"{json_obj_sha1}{path.stem}"
            if path.stem == path.name  # only extension given, no filestem
            else f"{path.stem}_{json_obj_sha1}{''.join(path.suffixes)}"
        )
        new_path = AnyPath(path.parts[0])
        for p in path.parts[1:-1]:
            new_path = new_path / p
        path = new_path / filename_sha1

    with path.open("w") as fp:
        fp.write(json_obj)

    logger.debug(f"Finished writing {str(path)}")
    return path


def write_png(obj: np.ndarray, path: AnyPath):
    with path.open("wb") as fp:
        Image.fromarray(obj).save(fp, format="png")
    logger.debug(f"Finished writing {str(path)}")
    return path


def write_npz(obj: Dict[str, np.ndarray], path: AnyPath):
    with path.open("wb") as fp:
        np.savez_compressed(fp, **obj)
    logger.debug(f"Finished writing {str(path)}")
    return path


def relative_path(path: AnyPath, start: AnyPath) -> AnyPath:
    result = os.path.relpath(path=str(path), start=str(start))

    return AnyPath(result)
