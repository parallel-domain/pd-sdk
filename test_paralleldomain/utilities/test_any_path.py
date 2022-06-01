from pathlib import Path
from tempfile import TemporaryDirectory

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


def test_relative():
    with TemporaryDirectory() as temp_dir:
        path_1 = AnyPath(temp_dir) / "test_1"
        path_2 = AnyPath(temp_dir) / "test_2"
        assert isinstance(path_1, AnyPath)
        assert isinstance(path_2, AnyPath)
        assert path_1.relative_to(path_2)._backend == AnyPath(r"..\test_1")._backend
