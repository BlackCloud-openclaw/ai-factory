import json
from collections import defaultdict
import statistics

# 使用绝对路径
json_path = '/home/data/projects/ai_factory/experiments/phase5/formal/reports/results/llm_scores.json'

with open(json_path, 'r') as f:
    data = json.load(f)

# 过滤有效数据
valid = [d for d in data if d['brief_reason'] not in ['解析失败', 'API 错误']]

# 按 pair 和 condition 分组
pair_cond = defaultdict(lambda: defaultdict(list))
for d in valid:
    parts = d['scene_id'].split('_')
    pair = parts[0] + '_' + parts[1]
    cond = d['condition']
    scores = [d['spatial_score'], d['physical_score'], d['intentional_score'], 
              d['informational_score'], d['temporal_score'], d['narrative_dependency_score']]
    pair_cond[pair][cond].append(scores)

print('='*80)
print('Pair × Condition 分析')
print('='*80)

pairs = sorted(pair_cond.keys())
conds = ['baseline', 'C1', 'C2', 'C3', 'C4']

for pair in pairs:
    print(f'\n{pair}:')
    print(f'{"Condition":<12} {"N":<4} {"Spatial":<10} {"Physical":<10} {"Intentional":<10} {"Info":<10} {"Temporal":<10} {"ND":<10}')
    for cond in conds:
        if cond not in pair_cond[pair]:
            continue
        scores = pair_cond[pair][cond]
        n = len(scores)
        avg = [statistics.mean(s[i] for s in scores) for i in range(6)]
        print(f'{cond:<12} {n:<4} {avg[0]:<10.2f} {avg[1]:<10.2f} {avg[2]:<10.2f} {avg[3]:<10.2f} {avg[4]:<10.2f} {avg[5]:<10.2f}')

    # C1 vs baseline (Spatial)
    if 'baseline' in pair_cond[pair] and 'C1' in pair_cond[pair]:
        base_spatial = statistics.mean(s[0] for s in pair_cond[pair]['baseline'])
        c1_spatial = statistics.mean(s[0] for s in pair_cond[pair]['C1'])
        delta = c1_spatial - base_spatial
        print(f'  C1 vs baseline (Spatial): {delta:+.2f}')

    # C2 vs baseline (Physical)
    if 'baseline' in pair_cond[pair] and 'C2' in pair_cond[pair]:
        base_physical = statistics.mean(s[1] for s in pair_cond[pair]['baseline'])
        c2_physical = statistics.mean(s[1] for s in pair_cond[pair]['C2'])
        delta = c2_physical - base_physical
        print(f'  C2 vs baseline (Physical): {delta:+.2f}')