# src/writing/snapshot/runtime/__init__.py
"""
B3: Runtime Foundation
"""

from .constants import RUNTIME_RECORD_FORMAT_VERSION
from .exceptions import (
    SnapshotRuntimeError,
    SnapshotSerializationError,
    SnapshotStoreError,
    SnapshotNotFoundError,
)
from .id import SnapshotId
from .metadata import SnapshotMetadata
from .record import SnapshotRecord
from .record_builder import RecordBuilder
from .protocols import SnapshotSerializer, SnapshotStore
from .path_strategy import SnapshotPathStrategy, DefaultSnapshotPathStrategy
from .memory_store import MemorySnapshotStore
from .file_store import FileSnapshotStore
from .pipeline import RuntimePipeline

# B3.2: Compression
from .compression import (
    CompressionCodec,
    CompressionResolver,
    CompressionRegistry,
    IdentityCodec,
    GzipCodec,
    CompressionError,
    UnsupportedCompressionError,
    CompressionDataError,
    DuplicateCompressionCodecError,
)

# B3.2: Serializers
from .serializers import (
    SerializerError,
    UnsupportedSerializerError,
    DuplicateSerializerError,
    SerializerRegistry,
    JsonSerializer,
)

# B3.3: Version
from .version import SemanticVersion

# B3.3: Chunking
from .chunking import (
    Chunk,
    ChunkingStrategy,
    FixedChunkStrategy,
    ChunkLayout,
    Assembler,
    StreamingManifest,
    ManifestBuilder,
)

# B3.3: ChunkStore
from .chunk_store import (
    ChunkReader,
    ChunkWriter,
    ChunkStore,
    MemoryChunkStore,
    FileChunkStore,
)

# B3.3: Transport
from .transport import (
    Transport,
    RecordTransport,
    ChunkTransport,
)

from .models import RuntimeSnapshot

__all__ = [
    "RUNTIME_RECORD_FORMAT_VERSION",
    "SnapshotRuntimeError",
    "SnapshotSerializationError",
    "SnapshotStoreError",
    "SnapshotNotFoundError",
    "SnapshotId",
    "SnapshotMetadata",
    "SnapshotRecord",
    "RecordBuilder",
    "SnapshotSerializer",
    "SnapshotStore",
    "SnapshotPathStrategy",
    "DefaultSnapshotPathStrategy",
    "MemorySnapshotStore",
    "FileSnapshotStore",
    "RuntimePipeline",
    # Compression
    "CompressionCodec",
    "CompressionResolver",
    "CompressionRegistry",
    "IdentityCodec",
    "GzipCodec",
    "CompressionError",
    "UnsupportedCompressionError",
    "CompressionDataError",
    "DuplicateCompressionCodecError",
    # Serializers
    "SerializerError",
    "UnsupportedSerializerError",
    "DuplicateSerializerError",
    "SerializerRegistry",
    "JsonSerializer",
    # B3.3
    "SemanticVersion",
    "Chunk",
    "ChunkingStrategy",
    "FixedChunkStrategy",
    "ChunkLayout",
    "Assembler",
    "StreamingManifest",
    "ManifestBuilder",
    "ChunkReader",
    "ChunkWriter",
    "ChunkStore",
    "MemoryChunkStore",
    "FileChunkStore",
    "Transport",
    "RecordTransport",
    "ChunkTransport",
    "RuntimeSnapshot",
]


# B3.5: Streaming
from .streaming import (
    StreamingSerializer,
    StreamingCompressionCodec,
    StreamingChunker,
    StreamingPipeline,
    create_default_streaming_pipeline,
)

__all__ += [
    "StreamingSerializer",
    "StreamingCompressionCodec",
    "StreamingChunker",
    "StreamingPipeline",
    "create_default_streaming_pipeline",
]