# tests/unit/writing/audit/test_hook.py

import pytest
import asyncio
from src.writing.audit import audit_writer, MemoryPayloadResolver


def test_audit_writer_sync():
    resolver = MemoryPayloadResolver()

    @audit_writer(resolver=resolver)
    def generate(novel_id: str, volume: int, chapter: int, scene_idx: int):
        return {"result": "test"}

    result = generate("novel", 1, 1, 0)
    assert result["result"] == "test"


def test_audit_writer_async():
    resolver = MemoryPayloadResolver()

    @audit_writer(resolver=resolver)
    async def generate_async(novel_id: str, volume: int, chapter: int, scene_idx: int):
        await asyncio.sleep(0.01)
        return {"result": "async test"}

    result = asyncio.run(generate_async("novel", 1, 1, 0))
    assert result["result"] == "async test"