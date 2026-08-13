"""
测试 Batch Runner（使用 Mock）
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from experiments.phase12.corpus.loader import CorpusLoader
from experiments.phase12.corpus.regenerator import CorpusRegenerator
from scripts.regenerate_corpus_v2 import regenerate_corpus


@pytest.mark.asyncio
async def test_batch_runner_dry_run(tmp_path):
    """测试 dry-run 模式不写入文件"""
    # 准备一个最小 v1.1 目录（此处需要 mock）
    v1_path = tmp_path / "v1.1"
    v1_path.mkdir()
    v2_path = tmp_path / "v2.0"

    # 创建 mock corpus.yaml
    corpus_data = {
        "version": "1.1",
        "samples": [
            {"path": "runtime_state/corpus.test.001.yaml"},
        ]
    }
    import yaml
    with open(v1_path / "corpus.yaml", "w") as f:
        yaml.safe_dump(corpus_data, f)

    # 创建 mock sample file
    sample_data = {
        "id": "corpus.test.001",
        "version": "1.1",
        "category": "runtime_state",
        "failure_modes": ["runtime_state"],
        "difficulty": "medium",
        "scene_before": "test",
        "scene_after": None,
        "expected": {},
        "artifacts": {},
    }
    runtime_dir = v1_path / "runtime_state"
    runtime_dir.mkdir()
    with open(runtime_dir / "corpus.test.001.yaml", "w") as f:
        yaml.safe_dump(sample_data, f)

    # 运行 dry-run
    await regenerate_corpus(
        v1_1_path=v1_path,
        v2_0_path=v2_path,
        limit=1,
        dry_run=True,
    )

    # 验证 v2.0 目录未创建
    assert not v2_path.exists()