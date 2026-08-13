# tests/unit/snapshot/runtime/test_chunking.py

import pytest

from src.writing.snapshot.runtime import (
    ChunkLayout,
    SemanticVersion,
    SnapshotId,           # 添加这一行
)
from src.writing.snapshot.runtime.chunking import (
    FixedChunkStrategy,
    Assembler,
    StreamingManifest,
    ManifestBuilder,
    Chunk,
)
from src.writing.snapshot.runtime.chunk_store import MemoryChunkStore


class TestChunkLayout:
    def test_serialization_roundtrip(self):
        layout = ChunkLayout(
            algorithm="fixed",
            target_chunk_size=1024,
            parameters={"extra": "value"},
        )
        mapping = layout.to_mapping()
        restored = ChunkLayout.from_mapping(mapping)
        assert restored == layout

class TestStreamingManifest:
    def test_serialization_roundtrip(self):
        sid = SnapshotId.new()
        layout = ChunkLayout(algorithm="fixed", target_chunk_size=1024)
        manifest = StreamingManifest(
            manifest_version=SemanticVersion(1, 0),
            snapshot_id=sid,
            total_chunks=2,
            total_size=2048,
            compressed_size=1536,
            serializer_id="builtin.json",
            codec_id="builtin.gzip",
            layout=layout,
        )
        mapping = manifest.to_mapping()
        restored = StreamingManifest.from_mapping(mapping)
        assert restored == manifest