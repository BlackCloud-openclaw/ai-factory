# experiments/phase5/formal/test_one.py
import asyncio
from runner import FormalExperimentRunner, generate_all_pairs
from pathlib import Path

async def test():
    runner = FormalExperimentRunner(Path('./reports/raw'))
    pairs = generate_all_pairs()
    pair = pairs[0]  # pair_01
    # 只运行 baseline 第一次
    result = await runner.run_single(pair, "baseline", 0, None)
    print("="*60)
    print("生成的文本:")
    print(result.get("text", "无文本"))
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test())