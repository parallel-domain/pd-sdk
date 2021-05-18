from typing.io import IO
from cloudpathlib import CloudPath
from smart_open.smart_open_lib import open

AnyPath = CloudPath


def _open_wrap(self, mode, *args, **kwargs) -> IO:
    return open(uri=self._str, mode=mode, *args, **kwargs)


AnyPath.open = _open_wrap
