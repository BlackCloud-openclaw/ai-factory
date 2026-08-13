"""
Phase 12.2B: Corpus Exporter

将 EvaluationSnapshot 序列化为 Corpus YAML v2.0 格式。
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict
from datetime import datetime
from src.writing.evaluation import EvaluationSnapshot


class CorpusExporter:
    """
    将 EvaluationSnapshot 导出为 Corpus YAML 样本。
    """

    def __init__(
        self,
        output_dir: Path,
        version: str = "2.0",
        default_category: str = "runtime_state",
        default_difficulty: str = "medium",
    ):
        self.output_dir = Path(output_dir)
        self.version = version
        self.default_category = default_category
        self.default_difficulty = default_difficulty
        # 延迟创建目录，直到 export 时

    def export(
        self,
        snapshot: EvaluationSnapshot,
        category: Optional[str] = None,
        failure_modes: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> Path:
        """导出单个 Snapshot 为 YAML 文件"""
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        category = category or self.default_category
        failure_modes = failure_modes or [category]
        difficulty = difficulty or self.default_difficulty
        sample_id = sample_id or f"corpus.auto.{snapshot.snapshot_id.hex[:8]}"

        sample = {
            "id": sample_id,
            "version": self.version,
            "category": category,
            "failure_modes": failure_modes,
            "difficulty": difficulty,
            "language": "zh-CN",
            "scene_before": snapshot.scene_before,
            "scene_after": snapshot.scene_after,
            "source": "generated",
            "license": "internal",
            "tags": ["phase12.2b", "auto_generated"],
            "expected": {},
            "artifacts": {
                "runtime_metrics": snapshot.runtime_metrics.to_dict() if snapshot.runtime_metrics else None,
                "revision_result": snapshot.revision_result.to_dict() if snapshot.revision_result else None,
                "judge_context": snapshot.judge_context.to_dict() if snapshot.judge_context else None,
                "events": snapshot.artifacts.get("events", []),
            },
        }

        # 生成文件名
        filename = f"{sample_id}.yaml"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.safe_dump(sample, f, allow_unicode=True, sort_keys=False)

        return filepath

    def export_manifest(self, sample_paths: List[Path]) -> Path:
        """生成 v2.0 corpus.yaml manifest"""
        manifest = {
            "version": self.version,
            "created_at": datetime.now().isoformat(),
            "categories": sorted({p.parent.name for p in sample_paths}),
            "samples": [
                {"path": str(p.relative_to(self.output_dir))}
                for p in sorted(sample_paths)
            ],
            "total_samples": len(sample_paths),
        }
        manifest_path = self.output_dir / "corpus.yaml"
        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)
        return manifest_path

    def export_batch(
        self,
        snapshots: List[EvaluationSnapshot],
        categories: Optional[List[str]] = None,
    ) -> List[Path]:
        """批量导出"""
        if categories is None:
            categories = [self.default_category] * len(snapshots)
        paths = []
        for idx, snapshot in enumerate(snapshots):
            cat = categories[idx] if idx < len(categories) else self.default_category
            path = self.export(snapshot, category=cat)
            paths.append(path)
        return paths