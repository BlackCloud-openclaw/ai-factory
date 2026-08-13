# src/writing/snapshot/__init__.py

from .models import *
from .serializer import JsonSerializer, Serializer
from .writer import SnapshotWriter
from .loader import SnapshotLoader
from .encoder import SnapshotEncoder
from .decoder import SnapshotDecoder
from .canonical_json import dumps

# ⭐ 关键：导入 encoders 子模块，触发注册
from . import encoders   # <-- 添加这一行

# ========== 兼容性：重新导出旧版 SnapshotManager ==========
# SnapshotManager 定义在 src/writing/snapshot.py（模块），
# 此处重新导出，使 from src.writing.snapshot_manager import SnapshotManager 可用
# ============================================================

__all__ = [
    "PipelineSnapshot",
    "SnapshotIdentity",
    "SnapshotManifest",
    "SnapshotMetadata",
    "JsonSerializer",
    "Serializer",
    "SnapshotWriter",
    "SnapshotLoader",
    "SnapshotEncoder",
    "SnapshotDecoder",
    "dumps",
]