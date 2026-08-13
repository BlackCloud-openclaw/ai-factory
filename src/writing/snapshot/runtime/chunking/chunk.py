# src/writing/snapshot/runtime/chunking/chunk.py
"""
B3.3: Chunk — 最小传输单元
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """数据块，Transport 层的最小传输单元。"""

    chunk_id: int       # 块标识符（不假定连续）
    payload: bytes      # 块数据（已压缩或未压缩）