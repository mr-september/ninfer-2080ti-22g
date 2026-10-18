"""Offset I/O and page-cache control for large offline transfers, on Linux and Windows.

Linux keeps whole artifacts out of the page cache with posix_fadvise and uses pread/pwrite. Windows has
neither, so reads and writes seek under a lock and the cache hints do nothing there.
"""

from __future__ import annotations

import os
import mmap
import threading

IO_CHUNK_BYTES = 8 * 1024 * 1024
WRITEBACK_BYTES = 64 * 1024 * 1024
_PAGE_BYTES = mmap.ALLOCATIONGRANULARITY if os.name == "nt" else mmap.PAGESIZE
_BINARY = getattr(os, "O_BINARY", 0)
_HAS_FADVISE = hasattr(os, "posix_fadvise")
_HAS_PREAD = hasattr(os, "pread")
_SEEK_LOCK = threading.Lock()
_MAX_IO_BYTES = 1 << 30  # Windows read/write take an int count


def open_read(path) -> int:
    return os.open(path, os.O_RDONLY | _BINARY)


def pread(fd: int, count: int, offset: int) -> bytes:
    """Read up to count bytes at offset; fewer only at end of file."""
    if _HAS_PREAD:
        return os.pread(fd, count, offset)
    chunks = []
    with _SEEK_LOCK:
        os.lseek(fd, offset, os.SEEK_SET)
        while count > 0:
            data = os.read(fd, min(count, _MAX_IO_BYTES))
            if not data:
                break
            chunks.append(data)
            count -= len(data)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def pwrite(fd: int, data, offset: int) -> int:
    if hasattr(os, "pwrite"):
        return os.pwrite(fd, data, offset)
    with _SEEK_LOCK:
        os.lseek(fd, offset, os.SEEK_SET)
        return os.write(fd, data)


def datasync(fd: int) -> None:
    getattr(os, "fdatasync", os.fsync)(fd)


def discard_cached_pages(fd: int, offset: int = 0, count: int | None = None) -> None:
    if not _HAS_FADVISE:
        return
    if count is None:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    elif count > 0:
        begin = offset // _PAGE_BYTES * _PAGE_BYTES
        end = (offset + count + _PAGE_BYTES - 1) // _PAGE_BYTES * _PAGE_BYTES
        os.posix_fadvise(fd, begin, end - begin, os.POSIX_FADV_DONTNEED)


class Writeback:
    """Bound dirty output across all open shards; release clean pages after writeback."""

    def __init__(self) -> None:
        self._bytes = 0
        self._fds: set[int] = set()

    def written(self, fd: int, count: int) -> None:
        self._fds.add(fd)
        self._bytes += count
        if self._bytes >= WRITEBACK_BYTES:
            self.flush()

    def flush(self) -> None:
        for fd in self._fds:
            datasync(fd)
            discard_cached_pages(fd)
        self._fds.clear()
        self._bytes = 0
