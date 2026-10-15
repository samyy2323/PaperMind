"""File storage abstraction.

All file access in the app goes through a StorageBackend, addressed by *keys* (posix-style relative
paths such as ``answer_sheets/12/page_001.jpg``). To move to S3-compatible storage, implement
``StorageBackend`` in a new module and return it from ``get_storage``.
"""
from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from app.core.config import get_settings


class StorageBackend(ABC):
    @abstractmethod
    def save(self, key: str, data: bytes) -> str: ...

    @abstractmethod
    def read(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None: ...

    @abstractmethod
    @contextmanager
    def local_path(self, key: str) -> Iterator[Path]:
        """Yield a filesystem path for the object (S3 backends would download to a temp file)."""


class LocalStorage(StorageBackend):
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError(f"Key escapes storage root: {key}")
        return p

    def save(self, key: str, data: bytes) -> str:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return key

    def read(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete_prefix(self, prefix: str) -> None:
        p = self._path(prefix)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()

    @contextmanager
    def local_path(self, key: str) -> Iterator[Path]:
        yield self._path(key)


@lru_cache
def _cached(root: str) -> StorageBackend:
    return LocalStorage(Path(root))


def get_storage() -> StorageBackend:
    return _cached(str(get_settings().STORAGE_ROOT))
