# we import from here just in case we change away from cloud lib in the future
# then we just need to replace the AnyPath Reference
from pathlib import Path
from typing import Callable, Union

from cloudpathlib import AnyPath as CloudLibAnyPath
from cloudpathlib import CloudPath

AnyPath: Callable[[Union[str, CloudPath, Path]], Union[CloudPath, Path]] = CloudLibAnyPath
