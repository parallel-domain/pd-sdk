import hashlib
import json
import logging
import os
from typing import Dict, List, Union

import cv2
import numpy as np

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
        fp.write(
            cv2.imencode(
                ext=".png",
                img=cv2.cvtColor(
                    src=obj,
                    code=cv2.COLOR_RGBA2BGRA,
                ),
            )[1].tobytes()
        )
    logger.debug(f"Finished writing {str(path)}")
    return path


def read_png(path: AnyPath) -> np.ndarray:
    with path.open(mode="rb") as fp:
        image_data = cv2.cvtColor(
            src=cv2.imdecode(
                buf=np.frombuffer(fp.read(), np.uint8),
                flags=cv2.IMREAD_UNCHANGED,
            ),
            code=cv2.COLOR_BGRA2RGBA,
        )
    return image_data


def write_npz(obj: Dict[str, np.ndarray], path: AnyPath):
    with path.open("wb") as fp:
        np.savez_compressed(fp, **obj)
    logger.debug(f"Finished writing {str(path)}")
    return path


def relative_path(path: AnyPath, start: AnyPath) -> AnyPath:
    result = os.path.relpath(path=str(path), start=str(start))

    return AnyPath(result)
