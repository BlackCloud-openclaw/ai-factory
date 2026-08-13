# src/writing/snapshot/migration/schema.py
"""
B2: Schema Runtime Configuration

定义当前系统支持的 Schema 版本，属于 Runtime Configuration，不属于 Version Model。
"""

from .version import SchemaVersion

# 当前系统支持的最新 Schema 版本
CURRENT_SCHEMA_VERSION = SchemaVersion(2, 0)

# 最低支持的 Snapshot 版本（加载时低于此版本应拒绝）
MINIMUM_SUPPORTED_VERSION = SchemaVersion(1, 0)