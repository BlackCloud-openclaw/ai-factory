# src/writing/snapshot/runtime/chunking/__init__.py
"""
B3.3: Chunking 模块
"""

from .chunk import Chunk
from .strategy import ChunkingStrategy
from .fixed import FixedChunkStrategy
from .layout import ChunkLayout
from .assembler import Assembler
from .manifest import StreamingManifest, ManifestBuilder

__all__ = [
    "Chunk",
    "ChunkingStrategy",
    "FixedChunkStrategy",
    "ChunkLayout",
    "Assembler",
    "StreamingManifest",
    "ManifestBuilder",
]