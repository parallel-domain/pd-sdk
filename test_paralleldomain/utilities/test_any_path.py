from pathlib import Path
from tempfile import TemporaryDirectory

from cloudpathlib import CloudPath

from paralleldomain.utilities.any_path import AnyPath


def test_resolving():
    with TemporaryDirectory() as temp_dir:
        path = AnyPath(temp_dir)
        assert path.exists()
        assert isinstance(path, Path)
        path = AnyPath("s3://paralleldomain-testing/")
        assert isinstance(path, CloudPath)
