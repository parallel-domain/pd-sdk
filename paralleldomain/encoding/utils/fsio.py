import json
from typing import Dict, List, Union

import numpy as np
from PIL import Image

from paralleldomain.utilities.any_path import AnyPath


def write_json(obj: Union[Dict, List], path: AnyPath):
    with path.open("w") as fp:
        json.dump(obj, fp, indent=2)


def write_png(obj: np.ndarray, path: AnyPath):
    with path.open("wb") as fp:
        Image.fromarray(obj).save(fp, format="png")
    print(f"Finished writing {str(path)}")


def write_npz(obj: Dict[str, np.ndarray], path: AnyPath):
    with path.open("wb") as fp:
        np.savez(fp, **obj)
