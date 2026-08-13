import json
from pathlib import Path
from typing import Protocol, runtime_checkable, Union, Any

from ..model import BenchmarkResult
from ..config.benchmark import SUMMARY_KEYS


@runtime_checkable
class BenchmarkReporter(Protocol):
    @property
    def content_type(self) -> str:
        ...

    def render(self, result: BenchmarkResult) -> str:
        ...

    def save(self, result: BenchmarkResult, path: Union[str, Path]) -> None:
        ...


class BaseReporter:
    def save(self, result: BenchmarkResult, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render(result))


class JsonReporter(BaseReporter):
    def __init__(self, indent: int = 2, ensure_ascii: bool = False):
        self._indent = indent
        self._ensure_ascii = ensure_ascii

    @property
    def content_type(self) -> str:
        return "application/json"

    def render(self, result: BenchmarkResult) -> str:
        return json.dumps(self._to_serializable(result.to_dict()), indent=self._indent, ensure_ascii=self._ensure_ascii)

    def _to_serializable(self, obj: Any) -> Any:
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        return obj


class MarkdownReporter(BaseReporter):
    @property
    def content_type(self) -> str:
        return "text/markdown"

    def render(self, result: BenchmarkResult) -> str:
        lines = [
            "# Benchmark Report",
            "",
            f"**Overall Score:** {result.overall_score:.3f}",
            f"**Version:** {result.metadata.get('benchmark_version', 'unknown')}",
            "",
            "## Metric Results",
            "",
            "| Metric | Score | State | Passed | Details |",
            "|--------|-------|-------|--------|---------|",
        ]
        for mr in result.metric_results:
            score = f"{mr.score:.3f}" if mr.score is not None else "N/A"
            state = mr.state.value
            passed = "✅" if mr.passed else "❌"
            details = self._format_details(mr.details)
            lines.append(f"| {mr.name} | {score} | {state} | {passed} | {details} |")

        if result.metadata:
            lines.append("")
            lines.append("## Metadata")
            lines.append("")
            for k, v in result.metadata.items():
                if k == "aggregator":
                    continue
                lines.append(f"- {k}: {v}")

        return "\n".join(lines)

    def _format_details(self, details: dict) -> str:
        parts = []
        for key in SUMMARY_KEYS:
            if key in details:
                value = details[key]
                if isinstance(value, list):
                    parts.append(f"{key}={len(value)} items")
                elif isinstance(value, dict):
                    parts.append(f"{key}={{...}}")
                elif isinstance(value, (int, float)):
                    if isinstance(value, float) and value % 1 != 0:
                        parts.append(f"{key}={value:.3f}")
                    else:
                        parts.append(f"{key}={value}")
                elif isinstance(value, str):
                    if len(value) > 30:
                        value = value[:27] + "..."
                    parts.append(f"{key}={value}")
        return ", ".join(parts) or "(empty)"