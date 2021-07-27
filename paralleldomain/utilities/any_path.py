# we import from here just in case we change away from cloud lib in the future
# then we just need to replace the AnyPath Reference
from pathlib import Path
from typing import Callable, Union

from cloudpathlib import AnyPath as CloudLibAnyPath
from cloudpathlib import CloudPath

open_old = CloudPath.open


def open_fix(self: CloudPath, *args, **kwargs):
    if "force_overwrite_from_cloud" not in kwargs:
        kwargs["force_overwrite_from_cloud"] = True

    if "force_overwrite_to_cloud" not in kwargs:
        kwargs["force_overwrite_to_cloud"] = True

    return open_old(self, *args, **kwargs)


CloudPath.open = open_fix

AnyPath: Callable[[Union[str, CloudPath, Path]], Union[CloudPath, Path]] = CloudLibAnyPath
