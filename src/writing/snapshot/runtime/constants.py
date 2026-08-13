# src/writing/snapshot/runtime/constants.py
"""
B3: Runtime 常量定义
"""

# Runtime Record 封装格式版本（非 SchemaVersion）
# 当 SnapshotRecord 的 Header 或封装方式发生变化时递增
RUNTIME_RECORD_FORMAT_VERSION = 1

# Magic Number
RUNTIME_RECORD_MAGIC = b"SNAP"