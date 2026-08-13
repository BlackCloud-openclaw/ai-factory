#!/usr/bin/env python3
"""
Phase 12.2B-3-2: Corpus Regeneration Batch Runner

加载 v1.1 Corpus，对每个样本调用 CorpusRegenerator，
生成 v2.0 YAML 文件。
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List
import argparse
import logging

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.writing.writer_factory import create_writer
from experiments.phase12.corpus.loader import CorpusLoader
from experiments.phase12.corpus.regenerator import CorpusRegenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def regenerate_corpus(
    v1_1_path: Path,
    v2_0_path: Path,
    limit: Optional[int] = None,
    category_filter: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """主再生流程"""
    # 1. 加载 v1.1 Corpus
    logger.info(f"Loading v1.1 Corpus from {v1_1_path}")
    loader = CorpusLoader()
    corpus = loader.load(v1_1_path / "corpus.yaml")
    samples = list(corpus.samples)

    if category_filter:
        samples = [
            s for s in samples
            if s.category == category_filter or (hasattr(s.category, "value") and s.category.value == category_filter)
        ]
        logger.info(f"Filtered to category: {category_filter}, {len(samples)} samples")

    if limit:
        samples = samples[:limit]
        logger.info(f"Limiting to {limit} samples")

    logger.info(f"Total samples to regenerate: {len(samples)}")

    if dry_run:
        logger.info("DRY RUN: Would regenerate the following samples:")
        for s in samples:
            logger.info(f"  - {s.id} ({s.category})")
        return

    # 2. 准备 Writer（使用适配器）
    adapter = create_writer()

    # 3. 准备 Regenerator
    regenerator = CorpusRegenerator(
        writer=adapter,
        output_dir=v2_0_path,
        version="2.0",
    )

    # 4. 逐个再生
    success_count = 0
    fail_count = 0
    output_paths = []

    for idx, sample in enumerate(samples, 1):
        logger.info(f"[{idx}/{len(samples)}] Regenerating sample: {sample.id}")
        try:
            result = await regenerator.regenerate_sample(
                sample=sample,
                category=sample.category,
                
                
                
            )
            if result.success:
                success_count += 1
                output_paths.append(result.output_path)
                logger.info(f"  ✅ Generated: {result.output_path}")
            else:
                fail_count += 1
                logger.error(f"  ❌ Failed: {result.error}")
        except Exception as e:
            fail_count += 1
            logger.error(f"  ❌ Exception: {e}")

    # 5. 生成 manifest
    if output_paths:
        manifest_path = regenerator._exporter.export_manifest(output_paths)
        logger.info(f"Manifest generated: {manifest_path}")

    logger.info(f"Regeneration complete: {success_count} succeeded, {fail_count} failed")
    if fail_count == 0:
        logger.info("✅ All samples regenerated successfully.")


def main():
    parser = argparse.ArgumentParser(description="Regenerate Corpus v2.0")
    parser.add_argument(
        "--v1-path",
        type=Path,
        default="experiments/phase12/corpus/v1.0",
        help="Path to v1.1 Corpus directory (default: experiments/phase12/corpus/v1.1)"
    )
    parser.add_argument(
        "--v2-path",
        type=Path,
        default="experiments/phase12/corpus/v2.0",
        help="Path to output v2.0 Corpus directory (default: experiments/phase12/corpus/v2.0)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to regenerate (for testing)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by category (e.g., runtime_state)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print samples that would be regenerated without executing"
    )
    args = parser.parse_args()

    asyncio.run(regenerate_corpus(
        v1_1_path=args.v1_path,
        v2_0_path=args.v2_path,
        limit=args.limit,
        category_filter=args.category,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()