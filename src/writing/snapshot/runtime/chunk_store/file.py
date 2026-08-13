# src/writing/snapshot/runtime/chunk_store/file.py
"""
B3.3: FileChunkStore — 文件系统分块存储
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from ..id import SnapshotId
from ..exceptions import SnapshotNotFoundError, SnapshotStoreError
from .protocol import ChunkReader, ChunkWriter, ChunkStore

if TYPE_CHECKING:
    from ..chunking import Chunk, StreamingManifest


def _chunk_path(base_dir: Path, snapshot_id: SnapshotId, chunk_id: int) -> Path:
    return base_dir / str(snapshot_id.value) / f"chunk_{chunk_id:08d}.bin"


def _manifest_path(base_dir: Path, snapshot_id: SnapshotId) -> Path:
    return base_dir / str(snapshot_id.value) / "manifest.json"


class _FileChunkReader:
    def __init__(self, base_dir: Path, snapshot_id: SnapshotId):
        self._base_dir = base_dir
        self._snapshot_id = snapshot_id

    def list_chunks(self) -> Iterable[int]:
        snapshot_dir = self._base_dir / str(self._snapshot_id.value)
        if not snapshot_dir.exists():
            return []
        chunk_ids = []
        for path in snapshot_dir.glob("chunk_*.bin"):
            try:
                name = path.stem
                chunk_id = int(name.split("_")[1])
                chunk_ids.append(chunk_id)
            except (IndexError, ValueError):
                continue
        return sorted(chunk_ids)

    def read_chunk(self, chunk_id: int) -> Chunk:
        path = _chunk_path(self._base_dir, self._snapshot_id, chunk_id)
        if not path.exists():
            raise ValueError(f"Chunk {chunk_id} not found")
        try:
            data = path.read_bytes()
            return Chunk(chunk_id=chunk_id, payload=data)
        except OSError as e:
            raise SnapshotStoreError(f"Failed to read chunk {chunk_id}: {e}") from e


class _FileChunkWriter:
    def __init__(self, base_dir: Path, snapshot_id: SnapshotId):
        self._base_dir = base_dir
        self._snapshot_id = snapshot_id
        self._snapshot_dir = base_dir / str(snapshot_id.value)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

    def append(self, chunk: Chunk) -> None:
        path = _chunk_path(self._base_dir, self._snapshot_id, chunk.chunk_id)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp_path.write_bytes(chunk.payload)
            tmp_path.replace(path)
        except OSError as e:
            raise SnapshotStoreError(f"Failed to write chunk {chunk.chunk_id}: {e}") from e
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)


class FileChunkStore:
    def __init__(self, base_dir: Path):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def create_writer(self, snapshot_id: SnapshotId) -> ChunkWriter:
        return _FileChunkWriter(self._base_dir, snapshot_id)

    def create_reader(self, snapshot_id: SnapshotId) -> ChunkReader:
        return _FileChunkReader(self._base_dir, snapshot_id)

    def write_manifest(self, snapshot_id: SnapshotId, manifest: StreamingManifest) -> None:
        path = _manifest_path(self._base_dir, snapshot_id)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            data = json.dumps(manifest.to_mapping(), indent=2, sort_keys=True).encode("utf-8")
            tmp_path.write_bytes(data)
            tmp_path.replace(path)
        except OSError as e:
            raise SnapshotStoreError(f"Failed to write manifest: {e}") from e
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def read_manifest(self, snapshot_id: SnapshotId) -> StreamingManifest:
        path = _manifest_path(self._base_dir, snapshot_id)
        if not path.exists():
            raise SnapshotNotFoundError(f"Manifest not found: {snapshot_id}")
        try:
            data = json.loads(path.read_bytes())
            return StreamingManifest.from_mapping(data)
        except (OSError, json.JSONDecodeError) as e:
            raise SnapshotStoreError(f"Failed to read manifest: {e}") from e

    def delete(self, snapshot_id: SnapshotId) -> None:
        snapshot_dir = self._base_dir / str(snapshot_id.value)
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)