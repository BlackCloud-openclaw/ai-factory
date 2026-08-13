import pytest
import yaml
from pathlib import Path
from datetime import datetime
from types import MappingProxyType

from experiments.phase12.corpus import (
    CorpusLoader,
    Corpus,
    CorpusMetadata,
    CorpusSample,
    CorpusArtifacts,
    ExpectedResult,
    FailureMode,
    Difficulty,
    ExpectationType,
    RuntimeArtifactAdapter,
    ContextFactory,
    JudgeContextFactory,
)

@pytest.fixture
def sample_yaml(tmp_path):
    """创建临时样本 YAML 文件"""
    content = """
id: corpus.test.001
version: "1.0"
category: scene_transition
failure_modes:
  - scene_transition
difficulty: medium
language: zh-CN

scene_before: "前一场景"
scene_after: "当前场景"

source: "test"
license: "internal"

expected:
  continuity:
    type: range
    min: 0.2
    max: 0.4

artifacts:
  runtime_metrics:
    retry_count: 1
"""
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


@pytest.fixture
def manifest_path(tmp_path):
    """创建 Manifest 文件"""
    manifest_content = """
version: "1.0"
created_at: "2026-07-23T00:00:00"
categories:
  - scene_transition
samples:
  - path: scene_transition/test.yaml
"""
    manifest_path = tmp_path / "corpus.yaml"
    manifest_path.write_text(manifest_content, encoding="utf-8")
    return manifest_path


def test_loader_load_sample(sample_yaml):
    loader = CorpusLoader()
    sample = loader.load_sample(sample_yaml)

    assert sample.id == "corpus.test.001"
    assert sample.version == "1.0"
    assert sample.category == FailureMode.SCENE_TRANSITION
    assert sample.difficulty == Difficulty.MEDIUM
    assert sample.language == "zh-CN"
    assert sample.scene_before == "前一场景"
    assert sample.scene_after == "当前场景"
    assert "continuity" in sample.expected
    assert sample.expected["continuity"].expectation_type == ExpectationType.RANGE
    assert sample.expected["continuity"].minimum == 0.2
    assert sample.expected["continuity"].maximum == 0.4
    assert sample.artifacts.runtime_metrics == {"retry_count": 1}


def test_loader_load_corpus(manifest_path, tmp_path):
    # 创建样本文件
    sample_content = """
id: corpus.test.002
version: "1.0"
category: scene_transition
failure_modes:
  - scene_transition
  - character_state
difficulty: easy
language: zh-CN
scene_before: ""
scene_after: ""
source: "test"
license: "internal"
expected:
  continuity:
    type: range
    min: 0.3
    max: 0.5
"""
    sample_dir = tmp_path / "scene_transition"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "test.yaml"
    sample_path.write_text(sample_content, encoding="utf-8")

    loader = CorpusLoader()
    corpus = loader.load(tmp_path)

    assert isinstance(corpus, Corpus)
    assert len(corpus.samples) == 1
    assert corpus.metadata.total_samples == 1
    assert corpus.samples[0].id == "corpus.test.002"


def test_loader_unsupported_version(tmp_path):
    """测试不支持的版本应报错"""
    manifest = {
        "version": "2.0",
        "created_at": "2026-07-23T00:00:00",
        "categories": [],
        "samples": [],
    }
    index_path = tmp_path / "corpus.yaml"
    with open(index_path, "w") as f:
        yaml.dump(manifest, f)

    loader = CorpusLoader()
    with pytest.raises(ValueError, match="Unsupported corpus version"):
        loader.load(tmp_path)


def test_expected_result_matches():
    # Exact
    exact = ExpectedResult("test", ExpectationType.EXACT, exact=0.5)
    assert exact.matches(0.5) is True
    assert exact.matches(0.51) is False

    # Range
    range_ = ExpectedResult("test", ExpectationType.RANGE, minimum=0.2, maximum=0.4)
    assert range_.matches(0.3) is True
    assert range_.matches(0.1) is False
    assert range_.matches(0.5) is False

    # Boolean
    bool_ = ExpectedResult("test", ExpectationType.BOOLEAN, boolean=True)
    assert bool_.matches(1.0) is True
    assert bool_.matches(0.0) is False

    # Tolerance
    exact_with_tolerance = ExpectedResult("test", ExpectationType.EXACT, exact=0.70, tolerance=0.02)
    assert exact_with_tolerance.matches(0.71) is True
    assert exact_with_tolerance.matches(0.69) is True
    assert exact_with_tolerance.matches(0.73) is False

    # None
    assert exact.matches(None) is False

    # CUSTOM 应抛异常
    custom = ExpectedResult("test", ExpectationType.CUSTOM)
    with pytest.raises(NotImplementedError):
        custom.matches(0.5)


def test_corpus_filter(manifest_path, tmp_path):
    # 创建样本
    sample_content1 = """
id: corpus.test.003
version: "1.0"
category: scene_transition
failure_modes:
  - scene_transition
difficulty: easy
language: zh-CN
scene_before: ""
scene_after: ""
source: "test"
license: "internal"
expected: {}
"""
    sample_content2 = """
id: corpus.test.004
version: "1.0"
category: character_state
failure_modes:
  - character_state
difficulty: hard
language: zh-CN
scene_before: ""
scene_after: ""
source: "test"
license: "internal"
expected: {}
"""
    sample_dir1 = tmp_path / "scene_transition"
    sample_dir1.mkdir(parents=True, exist_ok=True)
    sample_path1 = sample_dir1 / "test1.yaml"
    sample_path1.write_text(sample_content1, encoding="utf-8")

    sample_dir2 = tmp_path / "character_state"
    sample_dir2.mkdir(parents=True, exist_ok=True)
    sample_path2 = sample_dir2 / "test2.yaml"
    sample_path2.write_text(sample_content2, encoding="utf-8")

    # 更新 Manifest
    manifest_data = {
        "version": "1.0",
        "created_at": "2026-07-23T00:00:00",
        "categories": ["scene_transition", "character_state"],
        "samples": [
            {"path": "scene_transition/test1.yaml"},
            {"path": "character_state/test2.yaml"},
        ],
    }
    with open(tmp_path / "corpus.yaml", "w") as f:
        yaml.dump(manifest_data, f)

    loader = CorpusLoader()
    corpus = loader.load(tmp_path)

    filtered = corpus.filter_by_difficulty(Difficulty.EASY)
    assert len(filtered.samples) == 1
    assert filtered.samples[0].id == "corpus.test.003"

    filtered = corpus.filter_by_category(FailureMode.CHARACTER_STATE)
    assert len(filtered.samples) == 1
    assert filtered.samples[0].id == "corpus.test.004"


def test_corpus_metadata_immutable():
    """验证 failure_mode_distribution 确实是不可变的 MappingProxyType"""
    from types import MappingProxyType

    # 创建一个简单的样本
    sample = CorpusSample(
        id="test",
        version="1.0",
        category=FailureMode.SCENE_TRANSITION,
        failure_modes=(FailureMode.SCENE_TRANSITION,),
        difficulty=Difficulty.EASY,
        language="zh-CN",
        scene_before="",
        scene_after="",
        draft_before=None,
        draft_after=None,
        expected={},
        artifacts=CorpusArtifacts(),
        source="test",
        license="internal",
        tags=(),
    )

    metadata = CorpusMetadata.compute(
        samples=[sample],
        version="1.0",
        created_at=datetime.now(),
        categories=("scene_transition",),
    )

    assert isinstance(metadata.failure_mode_distribution, MappingProxyType)

    # 尝试修改应抛出异常
    with pytest.raises(TypeError):
        metadata.failure_mode_distribution["new_key"] = 1


def test_tag_filter_mode():
    """测试 tag 过滤的 any/all 模式"""
    sample1 = CorpusSample(
        id="test1",
        version="1.0",
        category=FailureMode.SCENE_TRANSITION,
        failure_modes=(FailureMode.SCENE_TRANSITION,),
        difficulty=Difficulty.EASY,
        language="zh-CN",
        scene_before="",
        scene_after="",
        draft_before=None,
        draft_after=None,
        expected={},
        artifacts=CorpusArtifacts(),
        source="test",
        license="internal",
        tags=("tag1", "tag2"),
    )
    sample2 = CorpusSample(
        id="test2",
        version="1.0",
        category=FailureMode.CHARACTER_STATE,
        failure_modes=(FailureMode.CHARACTER_STATE,),
        difficulty=Difficulty.MEDIUM,
        language="zh-CN",
        scene_before="",
        scene_after="",
        draft_before=None,
        draft_after=None,
        expected={},
        artifacts=CorpusArtifacts(),
        source="test",
        license="internal",
        tags=("tag2", "tag3"),
    )

    metadata = CorpusMetadata.compute(
        samples=[sample1, sample2],
        version="1.0",
        created_at=datetime.now(),
        categories=("scene_transition", "character_state"),
    )
    corpus = Corpus(metadata=metadata, samples=(sample1, sample2))

    # any 模式
    filtered_any = corpus.filter_by_tags({"tag1"}, mode="any")
    assert len(filtered_any.samples) == 1
    assert filtered_any.samples[0].id == "test1"

    # all 模式
    filtered_all = corpus.filter_by_tags({"tag2", "tag3"}, mode="all")
    assert len(filtered_all.samples) == 1
    assert filtered_all.samples[0].id == "test2"

    # 无效模式
    with pytest.raises(ValueError, match="Unknown tag filter mode"):
        corpus.filter_by_tags({"tag1"}, mode="invalid")