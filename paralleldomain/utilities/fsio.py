import hashlib
import logging
import os
import zipfile
from io import BytesIO
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import sleep
from typing import Dict, Iterable, List, Optional, TypeVar, Union

import cv2
import numpy as np
import ujson
from google.protobuf.descriptor_pool import DescriptorPool
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.message import Message
from PIL import Image

from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger("fsio")
TMessage = TypeVar("TMessage")


def write_json(obj: Union[Dict, List], path: AnyPath, append_sha1: bool = False):
    json_obj = ujson.dumps(obj, indent=2, escape_forward_slashes=False)

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


def read_json(path: AnyPath) -> Union[Dict, List]:
    with path.open("r") as fp:
        json_data = ujson.load(fp)

    return json_data


def read_json_str(json_str: str) -> Union[Dict, List]:
    json_data = ujson.loads(json_str)

    return json_data


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


def read_image(path: AnyPath, convert_to_rgb: bool = True, is_indexed=False) -> np.ndarray:
    with path.open(mode="rb") as fp:
        if is_indexed:
            pil_image = Image.open(BytesIO(fp.read()))
            return np.asarray(pil_image)
        image_data = cv2.imdecode(
            buf=np.frombuffer(fp.read(), np.uint8),
            flags=cv2.IMREAD_UNCHANGED,
        )
        if convert_to_rgb:
            color_convert_code = cv2.COLOR_BGR2RGB
            if image_data.shape[-1] == 4:
                color_convert_code = cv2.COLOR_BGRA2RGBA

            image_data = cv2.cvtColor(
                src=image_data,
                code=color_convert_code,
            )
    return image_data


def read_png(path: AnyPath) -> np.ndarray:
    return read_image(path=path, convert_to_rgb=True)


def write_npz(obj: Dict[str, np.ndarray], path: AnyPath):
    with path.open("wb") as fp:
        np.savez_compressed(fp, **obj)
    logger.debug(f"Finished writing {str(path)}")
    return path


def read_npz(
    path: AnyPath, files: Optional[Union[str, List[str]]] = None, max_retries: int = 3
) -> Union[
    Dict[str, Union[np.ndarray, Iterable, int, float, tuple, dict]],
    Union[np.ndarray, Iterable, int, float, tuple, dict],
]:
    if isinstance(files, str):
        files = [files]

    def read_npz_results(local_path: AnyPath) -> Dict[str, Union[np.ndarray, Iterable, int, float, tuple, dict]]:
        result = {}
        with local_path.open(mode="rb") as fp:
            npz_data = np.load(fp)
            for f in files if files else npz_data.files:
                result[f] = npz_data[f]
        return result

    if path.is_cloud_path:
        tries = 0
        success = False
        temp_dir = TemporaryDirectory()
        while not success and tries < max_retries:
            local_path = AnyPath(temp_dir.name) / path.name
            tries += 1
            path.copy(target=local_path)
            try:
                result = read_npz_results(local_path=local_path)
                success = True
            except zipfile.BadZipFile as e:
                local_path.unlink(missing_ok=True)
                if tries >= max_retries:
                    temp_dir.cleanup()
                    raise e
                else:
                    secs = 2.0 ** (tries - 1)
                    logger.info(
                        f"Caught BadZipFile exception. This might be due to connection problems. "
                        f"{tries}. retry in {secs}s"
                    )
                    sleep(secs)
        temp_dir.cleanup()
    else:
        result = read_npz_results(local_path=path)
    return result if len(result) != 1 else list(result.values())[0]


def read_json_message(
    obj: TMessage, path: AnyPath, ignore_unknown_fields: bool = True, descriptor_pool: DescriptorPool = None
) -> TMessage:
    with path.open("r") as fp:
        json_data = ujson.load(fp)

    result = ParseDict(
        js_dict=json_data,
        message=obj,
        ignore_unknown_fields=ignore_unknown_fields,
        descriptor_pool=descriptor_pool,
    )

    return result


def read_binary_message(obj: TMessage, path: AnyPath) -> TMessage:
    with path.open("rb") as fp:
        obj.ParseFromString(fp.read())

    return obj


def read_message(
    obj: TMessage, path: AnyPath, ignore_unknown_fields: bool = True, descriptor_pool: DescriptorPool = None
) -> TMessage:
    if str(path).endswith(".json"):
        return read_json_message(
            path=path, obj=obj, ignore_unknown_fields=ignore_unknown_fields, descriptor_pool=descriptor_pool
        )
    else:
        return read_binary_message(obj=obj, path=path)


def write_message(
    obj: Message,
    path: AnyPath,
    append_sha1: bool = False,
    including_default_value_fields: bool = True,
    preserving_proto_field_name: bool = True,
    use_integer_for_enums: bool = False,
    float_precision: int = None,
    descriptor_pool: DescriptorPool = None,
) -> TMessage:
    if str(path).endswith(".json"):
        return write_json_message(
            path=path,
            obj=obj,
            append_sha1=append_sha1,
            descriptor_pool=descriptor_pool,
            including_default_value_fields=including_default_value_fields,
            preserving_proto_field_name=preserving_proto_field_name,
            use_integer_for_enums=use_integer_for_enums,
            float_precision=float_precision,
        )
    else:
        return write_binary_message(obj=obj, path=path, append_sha1=append_sha1)


def write_json_message(
    obj: Message,
    path: AnyPath,
    append_sha1: bool = False,
    including_default_value_fields: bool = True,
    preserving_proto_field_name: bool = True,
    use_integer_for_enums: bool = False,
    float_precision: int = None,
    descriptor_pool: DescriptorPool = None,
) -> AnyPath:
    json_obj = ujson.dumps(
        MessageToDict(
            message=obj,
            including_default_value_fields=including_default_value_fields,
            preserving_proto_field_name=preserving_proto_field_name,
            use_integers_for_enums=use_integer_for_enums,
            descriptor_pool=descriptor_pool,
            float_precision=float_precision,
        ),
        indent=2,
        escape_forward_slashes=False,
    )

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


def write_binary_message(
    obj: Message,
    path: AnyPath,
    append_sha1: bool = False,
) -> AnyPath:
    enc_str = obj.SerializeToString(deterministic=True)

    if append_sha1:
        # noinspection InsecureHash
        json_obj_sha1 = hashlib.sha1(enc_str).hexdigest()
        filename_sha1 = (
            f"{json_obj_sha1}{path.stem}"
            if path.stem == path.name  # only extension given, no filestem
            else f"{path.stem}_{json_obj_sha1}{''.join(path.suffixes)}"
        )
        new_path = AnyPath(path.parts[0])
        for p in path.parts[1:-1]:
            new_path = new_path / p
        path = new_path / filename_sha1

    with path.open("wb") as fp:
        fp.write(enc_str)

    logger.debug(f"Finished writing {str(path)}")

    return path


def relative_path(path: AnyPath, start: AnyPath) -> AnyPath:
    result = os.path.relpath(path=str(path), start=str(start))

    return AnyPath(result)


def copy_file(source: AnyPath, target: AnyPath) -> AnyPath:
    source.copy(target=target)
    logger.debug(f"Finished copying from {str(source)} to {str(target)}")
    return target
