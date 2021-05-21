from tempfile import TemporaryDirectory

from cloudpathlib import CloudPath
from paralleldomain.utilities.any_path import AnyPath
from pathlib import Path


def test_resolving():
    with TemporaryDirectory() as temp_dir:
        path = AnyPath(temp_dir)
        assert path.exists()
        assert isinstance(path, Path)
        path = AnyPath("s3://paralleldomain-testing/")
        assert path.exists()
        assert isinstance(path, CloudPath)
