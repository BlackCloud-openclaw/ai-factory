# tests/unit/snapshot/runtime/remote/s3/test_gc_integration.py
"""
GC + S3 Adapter 集成测试
"""

import pytest
from unittest.mock import Mock

from src.writing.snapshot.runtime import SnapshotId
from src.writing.snapshot.runtime.incremental import MemoryVersionStore, VersionManifest
from src.writing.snapshot.runtime.remote.s3 import S3GCAdapter, S3Client, S3KeyLayout
from src.writing.snapshot.runtime.remote.gc import GarbageCollector, ChunkRef


class TestGCIntegration:
    def setup_method(self):
        self.version_store = MemoryVersionStore()
        self.client = Mock(spec=S3Client)
        self.layout = S3KeyLayout("snapshots/")
        self.adapter = S3GCAdapter(self.client, self.layout)
        self.gc = GarbageCollector(
            self.version_store,
            self.adapter,
            self.adapter,
        )

    def _create_manifest(self, sid: SnapshotId, parent_id: SnapshotId | None = None):
        manifest = VersionManifest(
            snapshot_id=sid,
            parent_id=parent_id,
            metadata={},
        )
        self.version_store.put(manifest)

    def test_dry_run_orphan_detection(self):
        sid = SnapshotId.new()
        self._create_manifest(sid)

        # 模拟 S3 列出所有对象：1 个有效 chunk + 1 个孤儿 chunk
        orphan_sid = SnapshotId.new()
        self.client.list_objects.return_value = [
            f"snapshots/{sid}/chunks/00000001.bin",
            f"snapshots/{orphan_sid}/chunks/00000099.bin",
        ]

        # 模拟 list_chunks 按 snapshot 返回（GarbageCollector 内部调用）
        # 注意：S3GCAdapter.list_chunks 会调用 list_objects 带前缀
        # 但我们直接 mock list_objects 的返回值
        # 为了测试简单，我们直接模拟 adapter 的行为
        # 但这里我们让 adapter 的 list_chunks 直接返回对应 snapshot 的 chunk
        # 由于 adapter 使用 self.client.list_objects，我们通过 side_effect 控制
        def list_objects_side_effect(prefix: str):
            if prefix == "snapshots/":
                # list_all_chunks 的调用
                return [
                    f"snapshots/{sid}/chunks/00000001.bin",
                    f"snapshots/{orphan_sid}/chunks/00000099.bin",
                ]
            elif prefix == f"snapshots/{sid}/chunks/":
                return [f"snapshots/{sid}/chunks/00000001.bin"]
            elif prefix == f"snapshots/{orphan_sid}/chunks/":
                return [f"snapshots/{orphan_sid}/chunks/00000099.bin"]
            return []

        self.client.list_objects.side_effect = list_objects_side_effect

        result = self.gc.collect(dry_run=True)

        # 孤儿 chunk 应被标记为待删除
        orphan_ref = ChunkRef(orphan_sid, 99)
        assert orphan_ref in result.deleted_chunks
        assert result.deleted_count == 1
        assert result.dry_run is True

    def test_actual_deletion(self):
        sid = SnapshotId.new()
        self._create_manifest(sid)

        orphan_sid = SnapshotId.new()
        def list_objects_side_effect(prefix: str):
            if prefix == "snapshots/":
                return [
                    f"snapshots/{sid}/chunks/00000001.bin",
                    f"snapshots/{orphan_sid}/chunks/00000099.bin",
                ]
            elif prefix == f"snapshots/{sid}/chunks/":
                return [f"snapshots/{sid}/chunks/00000001.bin"]
            elif prefix == f"snapshots/{orphan_sid}/chunks/":
                return [f"snapshots/{orphan_sid}/chunks/00000099.bin"]
            return []

        self.client.list_objects.side_effect = list_objects_side_effect

        result = self.gc.collect(dry_run=False)

        # 验证 delete_objects 被调用，且包含孤儿 chunk 键
        orphan_key = f"snapshots/{orphan_sid}/chunks/00000099.bin"
        self.client.delete_objects.assert_called_once()
        deleted_keys = self.client.delete_objects.call_args[0][0]
        assert orphan_key in deleted_keys
        assert result.deleted_count == 1
        assert result.dry_run is False