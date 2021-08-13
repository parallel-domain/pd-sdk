from pathlib import Path
from tempfile import TemporaryDirectory

from cloudpathlib import CloudPath

from paralleldomain.utilities.any_path import AnyPath, S3Path


def test_resolving():
    with TemporaryDirectory() as temp_dir:
        path = AnyPath(temp_dir)
        assert path.exists()
        assert isinstance(path._backend, Path)
        path = AnyPath("s3://paralleldomain-testing/")
        assert isinstance(path._backend, S3Path)


def test_concat():
    with TemporaryDirectory() as temp_dir:
        path = AnyPath(temp_dir) / "test"
        assert isinstance(path, AnyPath)
        path = AnyPath("s3://paralleldomain-testing/") / "test"
        assert isinstance(path, AnyPath)
