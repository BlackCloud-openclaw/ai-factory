"""
CorpusRepository：纯存储组件，不负责版本策略
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any

from ..corpus.models import CorpusSample


class CorpusRepository:
    """Corpus 存储管理，只负责 Manifest 读写和路径解析"""

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        self._manifest_path = base_dir / "corpus.yaml"

    def sample_path(self, sample: CorpusSample) -> Path:
        """样本存储路径（唯一来源）"""
        filename = sample.id if sample.id.endswith(".yaml") else f"{sample.id}.yaml"
        return self._base_dir / sample.category.value / filename

    def add_samples(self, samples: List[CorpusSample], version: str = "1.0") -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)

        manifest = self._load_manifest()
        existing_paths = {e["path"] for e in manifest.get("samples", [])}

        for sample in samples:
            rel_path = str(self.sample_path(sample).relative_to(self._base_dir))
            if rel_path not in existing_paths:
                manifest.setdefault("samples", []).append({"path": rel_path})

        manifest["version"] = version
        manifest["total_samples"] = len(manifest["samples"])

        with open(self._manifest_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)

    def load_manifest(self) -> Dict[str, Any]:
        return self._load_manifest()

    def _load_manifest(self) -> dict:
        if self._manifest_path.exists():
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {"version": "1.0", "samples": []}