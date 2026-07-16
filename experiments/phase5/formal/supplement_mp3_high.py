#!/usr/bin/env python3
"""
补充采集 MP3 High Gap 5 次，合并到现有数据中计算最终 TR
"""

import asyncio
import json
from collections import Counter
from pathlib import Path
from manipulation_check import ManipulationCheck


async def supplement_mp3_high():
    print("=" * 60)
    print("补充采集: MP3 High Gap (5 次)")
    print("=" * 60)
    
    check = ManipulationCheck()
    
    # 运行 5 次新的采样
    print("\n正在运行补充采样...")
    new_results = await check.run_gap("MP3", "high", samples=5)
    print(f"完成 {len(new_results)} 次采样")
    
    # 加载现有数据
    raw_path = Path("reports/manipulation_check/raw_data.json")
    with open(raw_path, 'r') as f:
        data = json.load(f)
    
    # 合并到 gap 数据中
    existing_gap = data['gap']
    
    # 移除旧的 MP3 high 数据（如果有，保留最新的）
    # 实际上，我们应该保留所有数据，但为了避免重复，我们只保留唯一的数据
    # 这里采用追加方式，不去重（因为每次采样的 raw_response 不同）
    for r in new_results:
        existing_gap.append(r)
    
    data['gap'] = existing_gap
    
    # 保存合并后的数据
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"数据已保存到: {raw_path}")
    
    # 重新计算 TR
    print("\n" + "=" * 60)
    print("MP3 High Gap TR 计算 (合并后)")
    print("=" * 60)
    
    all_mp3_high = [d for d in data['gap'] if d['mp_id'] == 'MP3' and d['gap_type'] == 'high']
    
    # 按采样时间排序（如果存在）
    top1s = []
    for d in all_mp3_high:
        if d.get('probabilities'):
            top1 = max(d['probabilities'], key=d['probabilities'].get)
            top1s.append(top1)
    
    n = len(top1s)
    counter = Counter(top1s)
    tr = counter.most_common(1)[0][1] / n if n > 0 else 0
    
    print(f"总样本数: {n}")
    print(f"Top-1 分布: {dict(counter)}")
    print(f"TR = {tr:.2f}")
    print(f"之前 (5次): 0.60")
    print(f"补充后 ({n}次): {tr:.2f}")
    
    if tr >= 0.80:
        print("\n结论: TR 回升至 0.80+，之前的 0.60 可能是随机波动。")
        print("→ MP3 High Gap 操纵不成立。")
    elif tr <= 0.70:
        print("\n结论: TR 稳定在 0.60-0.70，Gap 确实降低了预测稳定性。")
        print("→ MP3 High Gap 操纵有一定效果。")
    else:
        print(f"\n结论: TR = {tr:.2f}，处于中间区间，需要更多采样。")
    
    return tr, counter


if __name__ == "__main__":
    tr, dist = asyncio.run(supplement_mp3_high())