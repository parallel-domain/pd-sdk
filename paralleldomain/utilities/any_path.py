import abc
import os
from pathlib import Path
from typing import Union

from typing.io import IO
from cloudpathlib import CloudPath
from smart_open.smart_open_lib import open


AdjustedCloudPath = CloudPath

if os.environ.get("USESMARTOPEN", True):
    def _open_wrap(self, mode, *args, **kwargs) -> IO:
        return open(uri=self._str, mode=mode, *args, **kwargs)

    AdjustedCloudPath.open = _open_wrap


class AnyPath:
    def __new__(cls, path: str, *args, **kwargs):
        # Dispatch to subclass if base CloudPath
        if path.startswith("s3") or path.startswith("gs") or path.startswith("azure"):
            return AdjustedCloudPath(path, *args, **kwargs)
        return Path(path, *args, **kwargs)

