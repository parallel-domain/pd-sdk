import logging
import os
import shutil
import subprocess
from multiprocessing.pool import ThreadPool
from pathlib import Path, PurePath
from typing import List, Optional, Tuple, Union
from urllib.parse import urlparse

from s3path import S3Path

logger = logging.getLogger(__name__)


class AnyPath:
    def __init__(self, path: Union[str, "AnyPath"]):
        path = str(path)
        self._full_path = path
        parsed = urlparse(path)
        if parsed.scheme == "s3":
            self._backend = S3Path(path.replace("s3://", "/"))
        else:
            self._backend = Path(path)

    @staticmethod
    def _create_valid_any_path(new_path: Union[Path, S3Path]) -> "AnyPath":
        if isinstance(new_path, S3Path):
            if str(new_path).startswith("/"):
                new_path = str(new_path)[1:]
            return AnyPath(path=f"s3://{new_path}")
        else:
            return AnyPath(path=str(new_path))

    def __truediv__(self, other) -> "AnyPath":
        concat = self._backend / other
        return self._create_valid_any_path(new_path=concat)

    def __repr__(self):
        if isinstance(self._backend, S3Path):
            pth = self._backend
            if str(pth).startswith("/"):
                pth = str(pth)[1:]
            return f"s3://{pth}"
        else:
            return str(self._backend)

    def open(self, mode: str = "r", buffering: int = -1, encoding=None, errors=None, newline=None):
        """
        Open the file pointed by this path and return a file object, as
        the built-in open() function does.
        """
        return self._backend.open(mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline)

    def copytree(self, target: "AnyPath", max_num_threads: Optional[int] = None):
        target = AnyPath(str(target))
        if max_num_threads is None:
            max_num_threads = max(1, min(int(0.5 * os.cpu_count()), 10))

        def _copy(source: Path):
            if source.is_file():
                rel = source.relative_to(self._backend)
                to = target / rel
                logger.debug(f"copy {source} to {to}")
                to.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as source_file, to.open("wb") as to_file:
                    shutil.copyfileobj(source_file, to_file)

        with ThreadPool(max_num_threads) as pool:
            pool.map(_copy, (i for i in self._backend.rglob("*")))

    def relative_to(self, other: "AnyPath") -> "AnyPath":
        if isinstance(self._backend, type(other._backend)):
            rel_to = os.path.relpath(path=str(self), start=str(other))
            return AnyPath(path=str(rel_to))
        else:
            raise TypeError("Not possible to compare different backend types.")

    def copy(self, target: "AnyPath"):
        if self.is_cloud_path and target.is_cloud_path:
            command = ["aws", "s3", "cp", str(self), str(target), "--no-progress"]
            if self.is_dir():
                command += ["--recursive"]
            subprocess.call(command)
        else:
            with self.open("rb") as source_file, target.open("wb") as to_file:
                shutil.copyfileobj(source_file, to_file)

    @property
    def is_cloud_path(self) -> bool:
        return isinstance(self._backend, S3Path)

    @property
    def parent(self) -> "AnyPath":
        parent = self._backend.parent
        return self._create_valid_any_path(new_path=parent)

    @property
    def stem(self) -> str:
        return self._backend.stem

    @property
    def suffixes(self) -> List[str]:
        return self._backend.suffixes

    @property
    def suffix(self) -> str:
        return self._backend.suffix

    @property
    def name(self) -> str:
        return self._backend.name

    @property
    def parts(self) -> Tuple[str, ...]:
        if self.is_cloud_path:
            return tuple(["s3://"]) + self._backend.parts[1:]
        else:
            return self._backend.parts

    def as_posix(self) -> str:
        return self._backend.as_posix()

    def stat(self):
        """
        Returns information about this path (similarly to boto3's ObjectSummary).
        For compatibility with pathlib, the returned object some similar attributes like os.stat_result.
        The result is looked up at each call to this method
        """
        return self._backend.stat()

    def exists(self) -> bool:
        """
        Whether the path points to an existing Bucket, key or key prefix.
        """
        return self._backend.exists()

    def is_dir(self) -> bool:
        """
        Returns True if the path points to a Bucket or a key prefix, False if it points to a full key path.
        False is also returned if the path doesn’t exist.
        Other errors (such as permission errors) are propagated.
        """
        return self._backend.is_dir()

    def is_file(self) -> bool:
        """
        Returns True if the path points to a Bucket key, False if it points to Bucket or a key prefix.
        False is also returned if the path doesn’t exist.
        Other errors (such as permission errors) are propagated.
        """
        return self._backend.is_file()

    def iterdir(self):
        """
        When the path points to a Bucket or a key prefix, yield path objects of the directory contents
        """
        for path in self._backend.iterdir():
            yield self._create_valid_any_path(new_path=path)

    def glob(self, pattern: str):
        """
        Glob the given relative pattern in the Bucket / key prefix represented by this path,
        yielding all matching files (of any kind)
        """
        for path in self._backend.glob(pattern=pattern):
            yield self._create_valid_any_path(new_path=path)

    def rglob(self, pattern: str):
        """
        This is like calling S3Path.glob with "**/" added in front of the given relative pattern
        """
        for path in self._backend.rglob(pattern=pattern):
            yield self._create_valid_any_path(new_path=path)

    def owner(self):
        """
        Returns the name of the user owning the Bucket or key.
        Similarly to boto3's ObjectSummary owner attribute
        """
        return self._backend.owner()

    def rename(self, target: str):
        """
        Renames this file or Bucket / key prefix / key to the given target.
        If target exists and is a file, it will be replaced silently if the user has permission.
        If path is a key prefix, it will replace all the keys with the same prefix to the new target prefix.
        Target can be either a string or another S3Path object.
        """
        return self._backend.rename(target=str(target))

    def replace(self, target):
        """
        Renames this Bucket / key prefix / key to the given target.
        If target points to an existing Bucket / key prefix / key, it will be unconditionally replaced.
        """
        return self._backend.replace(target=str(target))

    def unlink(self, missing_ok: bool = False):
        """
        Remove this key from its bucket.
        """
        return self._backend.unlink(missing_ok=missing_ok)

    def rmdir(self):
        """
        Removes this Bucket / key prefix. The Bucket / key prefix must be empty
        """
        return self._backend.rmdir()

    def samefile(self, other_path: Union["AnyPath", Path, str]) -> bool:
        """
        Returns whether this path points to the same Bucket key as other_path,
        Which can be either a Path object, or a string
        """
        if isinstance(other_path, AnyPath):
            return self._backend.samefile(other_path=other_path._backend)
        return self._backend.samefile(other_path=str(other_path))

    def touch(self, mode=0o666, exist_ok: bool = True):
        """
        Creates a key at this given path.
        If the key already exists,
        the function succeeds if exist_ok is true (and its modification time is updated to the current time),
        otherwise FileExistsError is raised
        """
        return self._backend.touch(mode=mode, exist_ok=exist_ok)

    def mkdir(self, mode=0o777, parents=False, exist_ok=False):
        """
        Create a path bucket.
        AWS S3 Service doesn't support folders, therefore the mkdir method will only create the current bucket.
        If the bucket path already exists, FileExistsError is raised.

        If exist_ok is false (the default), FileExistsError is raised if the target Bucket already exists.
        If exist_ok is true, OSError exceptions will be ignored.

        if parents is false (the default), mkdir will create the bucket only if this is a Bucket path.
        if parents is true, mkdir will create the bucket even if the path have a Key path.

        mode argument is ignored.
        """
        return self._backend.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)

    def is_mount(self) -> bool:
        """
        AWS S3 Service doesn't have mounting feature, There for this method will always return False
        """
        return self._backend.is_mount()

    def is_symlink(self) -> bool:
        """
        AWS S3 Service doesn't have symlink feature, There for this method will always return False
        """
        return self._backend.is_symlink()

    def is_socket(self) -> bool:
        """
        AWS S3 Service doesn't have sockets feature, There for this method will always return False
        """
        return self._backend.is_socket()

    def is_fifo(self) -> bool:
        """
        AWS S3 Service doesn't have fifo feature, There for this method will always return False
        """
        return self._backend.is_fifo()
