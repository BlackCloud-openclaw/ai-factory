#!/usr/bin/env python
"""
命令行入口：运行提取 Pipeline
"""

import argparse
import sys
from pathlib import Path

from .config import ExtractionConfig
from .log_provider import LogFailureProvider
from .normalizer import FailureNormalizer
from .classifier import FailureClassifier
from .builder import CorpusSampleBuilder
from .validator import SchemaValidator
from .exporter import YamlExporter
from .repository import CorpusRepository
from .pipeline import ExtractionPipeline, SequentialSampleIdGenerator


def main():
    parser = argparse.ArgumentParser(description="Extract Gold Corpus from logs")
    parser.add_argument("--log", type=str, default="logs/ai_factory.log", help="Log file path")
    parser.add_argument("--max", type=int, default=10, help="Max records to extract")
    parser.add_argument("--output", type=str, default="experiments/phase12/corpus/v1.0", help="Output directory")
    parser.add_argument("--version", type=str, default="1.0", help="Corpus version")
    parser.add_argument("--patterns", type=str, nargs="+", help="Failure patterns")
    args = parser.parse_args()

    config = ExtractionConfig(
        log_paths=[Path(args.log)],
        max_records=args.max,
        output_dir=Path(args.output),
        corpus_version=args.version,
        failure_patterns=args.patterns or ExtractionConfig.default().failure_patterns,
    )

    # Composition Root
    provider = LogFailureProvider(
        log_paths=config.log_paths,
        failure_patterns=config.failure_patterns,
        max_records=config.max_records,
    )

    normalizer = FailureNormalizer()
    classifier = FailureClassifier()

    # expected_profiles
    expected_profiles = {
        "scene_transition": {
            "continuity": {"type": "range", "min": 0.3, "max": 0.5}
        },
        "character_state": {
            "character": {"type": "range", "min": 0.4, "max": 0.6}
        },
        "dialogue_quality": {
            "dialogue": {"type": "range", "min": 0.3, "max": 0.5}
        },
        "planning_execution": {
            "planning_coverage": {"type": "range", "min": 0.6, "max": 0.8}
        },
        "runtime_state": {
            "runtime_health": {"type": "range", "min": 0.5, "max": 0.7}
        },
    }
    builder = CorpusSampleBuilder(expected_profiles)

    validator = SchemaValidator()
    exporter = YamlExporter()
    repository = CorpusRepository(config.output_dir)
    id_generator = SequentialSampleIdGenerator()

    pipeline = ExtractionPipeline(
        provider=provider,
        normalizer=normalizer,
        classifier=classifier,
        builder=builder,
        id_generator=id_generator,
        repository=repository,
        validator=validator,
        exporter=exporter,
        config=config,
    )

    result = pipeline.run()

    print(f"\n{result}")
    if result.success:
        print(f"✅ Successfully extracted {result.exported} samples")
        for sample in result.samples:
            print(f"  - {sample.id}")
        sys.exit(0)
    else:
        print(f"❌ Extraction failed: {len(result.stats.errors)} errors")
        for err in result.stats.errors[:3]:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()