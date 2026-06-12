"""Utility functions for I/O operations."""

__all__ = [
    "PathLike",
    "assert_directory_exists",
    "cache_create",
    "resolve_path",
]

import json
import sys
import tarfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from importlib import metadata
from io import BytesIO
from pathlib import Path
from typing import IO, Any, Literal, TypeVar

import numpy as np
import zfpy
from diskcache import Cache

from cat_modeling.constants import CACHE_DIR

# NOTE(@vpsiena): override tarfile default record size
# to ensure minimal padding in tar archives
tarfile.RECORDSIZE = tarfile.BLOCKSIZE * 2

PathLike = Path | str


def resolve_path(root: PathLike) -> Path:
    """Resolve the given path."""
    return Path(root).expanduser().resolve()


def assert_directory_exists(root: PathLike) -> Path:
    """Assert that the given directory exists."""
    path = resolve_path(root)
    if not path.is_dir():
        msg = f"Directory does not exist: {root}"
        raise FileNotFoundError(msg)
    return path


def _si_size_bytes(gb: float, mb: float, kb: float, b: float) -> int:
    """Calculate the total size in bytes."""
    return int(sum(v * 1_000**d for d, v in enumerate((b, kb, mb, gb))))


def as_scope_name(obj: Any) -> str:  # noqa: ANN401
    """Get the scope name for the given object or class.

    Args:
        obj: The scope name, object or class to get the scope name for.

    Returns:
        The scope name of the object or class or given string.

    """
    if isinstance(obj, str):
        return obj

    if hasattr(obj, "__module__") and hasattr(obj, "__name__"):
        return f"{obj.__module__}.{obj.__name__}"
    if not isinstance(obj, type):
        return as_scope_name(obj.__class__)
    msg = "Cannot determine scope name."
    raise TypeError(msg)


def cache_create(
    scope: Any,  # noqa: ANN401
    root: PathLike | None = None,
    *,
    size_gb: float = 0.0,
    size_mb: float = 0.0,
) -> Cache:
    """Create specific isolated in scope disk cache.

    Args:
        scope: The scope name or object indicating the scope.
        root: The root directory for the cache.
        size_gb: The size of the cache in gigabytes.
        size_mb: The size of the cache in megabytes.

    Note:
        If all sizes are zero, the cache will default to 64MB.

    """
    scope = as_scope_name(scope)
    root = resolve_path(root or CACHE_DIR)
    if root == resolve_path(CACHE_DIR):
        root.mkdir(parents=True, exist_ok=True)
    root = assert_directory_exists(root)
    return Cache(
        directory=(root / scope).as_posix(),
        size_limit=_si_size_bytes(gb=size_gb, mb=size_mb, kb=0.0, b=0.0)
        or _si_size_bytes(gb=0.0, mb=64.0, kb=0.0, b=0.0),
    )


T = TypeVar("T")


class Codec[T](ABC):
    @abstractmethod
    def encode(self, data: T, buffer: IO[bytes]) -> int:
        raise NotImplementedError

    @abstractmethod
    def decode(self, buffer: IO[bytes]) -> T:
        raise NotImplementedError

    @property
    @abstractmethod
    def extension(self) -> str:
        raise NotImplementedError

    def filename(self, name: str) -> str:
        """Get the filename with the codec extension.

        Args:
            name: The base filename.

        Returns:
            The filename with the codec extension.

        """
        return f"{name}{self.extension}"


class JsonCodec(Codec[str | dict[str, Any] | list[Any]]):
    def __init__(
        self,
        *,
        compact: bool = False,
        ensure_ascii: bool = False,
        allow_nan: bool = False,
    ) -> None:
        """Initialize the JSON codec.

        Args:
            compact: Whether to use compact separators.
            ensure_ascii: Whether to ensure ASCII encoding.
            allow_nan: Whether to allow NaN values.

        """
        self._item_sep = "," if compact else ", "
        self._key_sep = ":" if compact else ": "
        self._ensure_ascii = ensure_ascii
        self._allow_nan = allow_nan

    @property
    def _dumps(self) -> Callable[[Any], str]:
        return json.JSONEncoder(
            ensure_ascii=self._ensure_ascii,
            allow_nan=self._allow_nan,
            sort_keys=False,
            separators=(self._item_sep, self._key_sep),
        ).encode

    def encode(self, data: str | dict[str, Any] | list[Any], buffer: IO[bytes]) -> int:
        position = buffer.tell()
        result = self._dumps(data).encode("utf-8")
        buffer.write(result)
        return buffer.tell() - position

    def decode(self, buffer: IO[bytes]) -> str | dict[str, Any] | list[Any]:
        return json.load(buffer)

    @property
    def extension(self) -> str:
        return ".json"


def _env_metadata() -> dict[str, str]:
    py = sys.version_info
    return {
        "python": f"{py.major}.{py.minor}.{py.micro}",
        "cat_modeling": metadata.version("cat_modeling"),
    }


class TarCodec(Codec[T]):
    _metadata_file_name = "_metadata.json"

    def __init__(
        self,
        *,
        compression: Literal["gz", "bz2", "xz"] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the TAR codec."""
        super().__init__()
        self._copression = compression
        self._write_mode = "w|" if compression is None else f"w|{compression}"
        self._read_mode = "r|" if compression is None else f"r|{compression}"
        self._metadata = (metadata or {}) | {"codec": self.__class__.__name__}

    @abstractmethod
    def encode_parts(self, data: T) -> Iterator[tuple[str, IO[bytes], int]]:
        raise NotImplementedError

    def encode(self, data: T, buffer: IO[bytes]) -> int:
        position = buffer.tell()
        with tarfile.open(fileobj=buffer, mode=self._write_mode) as archive:
            # write metadata file
            with BytesIO() as metadata_buffer:
                metadata_size = JsonCodec(compact=True).encode(
                    data=_env_metadata() | self._metadata,
                    buffer=metadata_buffer,
                )
                metadata_file_info = tarfile.TarInfo(name=self._metadata_file_name)
                metadata_file_info.size = metadata_size
                metadata_buffer.seek(0)
                archive.addfile(fileobj=metadata_buffer, tarinfo=metadata_file_info)

            # write content files
            for name, data_buffer, size in self.encode_parts(data):
                file_info = tarfile.TarInfo(name=name)
                file_info.size = size
                archive.addfile(fileobj=data_buffer, tarinfo=file_info)
        return buffer.tell() - position

    @abstractmethod
    def decode_parts(self, parts: Iterable[tuple[str, IO[bytes]]]) -> T:
        raise NotImplementedError

    @staticmethod
    def _assert_buffer(buffer: IO[bytes] | None) -> IO[bytes]:
        if buffer is None:
            msg = "Extracted file is None."
            raise ValueError(msg)
        return buffer

    def _reader(self, buffer: IO[bytes]) -> Iterator[tuple[str, IO[bytes]]]:
        with tarfile.open(fileobj=buffer, mode=self._read_mode) as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if member.name == self._metadata_file_name:
                    continue
                yield member.name, self._assert_buffer(archive.extractfile(member))

    def decode(self, buffer: IO[bytes]) -> T:
        return self.decode_parts(self._reader(buffer))

    @classmethod
    def metadata(cls, buffer: IO[bytes]) -> dict[str, str]:
        with tarfile.open(fileobj=buffer, mode="r|*") as archive:
            for member in archive:
                if member.name != cls._metadata_file_name:
                    continue
                metadata_buffer = cls._assert_buffer(archive.extractfile(member))
                result = JsonCodec().decode(metadata_buffer)
                if not isinstance(result, dict):
                    msg = "Metadata is not a dictionary."
                    raise TypeError(msg)
                return result
        msg = "Metadata file not found in archive."
        raise FileNotFoundError(msg)

    @property
    def extension(self) -> str:
        match self._copression:
            case "gz":
                return ".tar.gz"
            case "bz2":
                return ".tar.bz2"
            case "xz":
                return ".tar.xz"
            case None:
                return ".tar"
            case _:
                msg = f"Unsupported compression type: {self._copression}"
                raise NotImplementedError(msg)


_AnyArray = np.typing.NDArray[Any]


class NumpyCodec(Codec[dict[str, _AnyArray]]):
    def __init__(self, *, compression: bool) -> None:
        """Initialize the Numpy codec.

        Args:
            compression: Whether to use compression.

        """
        self._compression = compression

    def encode(self, data: dict[str, _AnyArray], buffer: IO[bytes]) -> int:
        position = buffer.tell()
        if self._compression:
            np.savez_compressed(buffer, allow_pickle=False, **data)
        else:
            np.savez(buffer, allow_pickle=False, **data)
        return buffer.tell() - position

    def decode(self, buffer: IO[bytes]) -> dict[str, _AnyArray]:
        # NOTE(@vpsiena): np.load requires a seekable buffer
        with (
            BytesIO(buffer.read()) as _buffer,
            np.load(_buffer, allow_pickle=False) as view,
        ):
            return dict(view)

    @property
    def extension(self) -> str:
        if self._compression:
            return ".npz"
        return ".npy"


class ZfpCodec(Codec[_AnyArray]):
    def __init__(self, *, tolerance: float = 0.0) -> None:
        """Initialize the ZFP codec.

        Args:
            tolerance: The absolute error tolerance.

        """
        if tolerance < 0.0:
            msg = "Tolerance must be non-negative."
            raise ValueError(msg)
        self._tolerance = tolerance

    def encode(self, data: _AnyArray, buffer: IO[bytes]) -> int:
        position = buffer.tell()
        compressed = zfpy.compress_numpy(data, tolerance=self._tolerance)
        buffer.write(compressed)
        return buffer.tell() - position

    def decode(self, buffer: IO[bytes]) -> _AnyArray:
        return zfpy.decompress_numpy(buffer.read())

    @property
    def extension(self) -> str:
        return ".zfp"


class DirectoryTarCodec(TarCodec[PathLike]):
    def __init__(
        self,
        *,
        root: PathLike,
        compression: Literal["gz", "bz2", "xz"] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(compression=compression, metadata=metadata)
        self.root = Path(root)
        self.root.mkdir(parents=False, exist_ok=True)
        if not self.root.is_dir():
            msg = f"Root path {self.root} is not a directory."
            raise NotADirectoryError(msg)

    def encode_parts(self, data: Path | str) -> Iterator[tuple[str, IO[bytes], int]]:
        data = Path(data)
        if not data.is_relative_to(self.root):
            msg = f"Path {data} is not relative to root {self.root}."
            raise ValueError(msg)
        for path in data.rglob("*"):
            if not path.is_file() or path.name.startswith("."):
                continue
            relative = path.relative_to(self.root).as_posix()
            with path.open("rb") as buffer:
                yield relative, buffer, path.stat().st_size

    def decode_parts(self, parts: Iterable[tuple[str, IO[bytes]]]) -> Path:
        for name, buffer in parts:
            output_path = self.root / name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("wb") as output_buffer:
                output_buffer.write(buffer.read())

        return self.root
