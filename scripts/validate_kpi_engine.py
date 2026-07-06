"""
KPI Engine 验证脚本
使用校准集（10 个场景）验证引擎输出与人工评分的一致性
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.writing.narrative_kpi import NarrativeKPIEngine


# ========== 人工评分基准（来自 Phase 2 校准标注） ==========
HUMAN_SCORES = {
    "CAL_EXP_01": {
        "dialogue": 1.0, "interaction": 1.0, "conflict": 1.5, "pressure": 1.0,
        "tension": 2.5, "relationship": 1.0, "goal": 1.0, "character": 1.0
    },
    "CAL_EXP_02": {
        "dialogue": 1.0, "interaction": 1.0, "conflict": 1.0, "pressure": 1.0,
        "tension": 1.5, "relationship": 1.0, "goal": 1.0, "character": 1.0
    },
    "CAL_DIA_01": {
        "dialogue": 4.0, "interaction": 3.5, "conflict": 3.0, "pressure": 2.5,
        "tension": 3.5, "relationship": 4.0, "goal": 2.0, "character": 3.0
    },
    "CAL_DIA_02": {
        "dialogue": 4.5, "interaction": 3.0, "conflict": 2.5, "pressure": 2.0,
        "tension": 4.0, "relationship": 3.0, "goal": 1.5, "character": 3.0
    },
    "CAL_REL_01": {
        "dialogue": 3.0, "interaction": 5.0, "conflict": 4.0, "pressure": 4.0,
        "tension": 4.5, "relationship": 5.0, "goal": 3.0, "character": 4.0
    },
    "CAL_REL_02": {
        "dialogue": 2.0, "interaction": 4.5, "conflict": 3.5, "pressure": 2.5,
        "tension": 3.5, "relationship": 4.0, "goal": 2.0, "character": 3.5
    },
    "CAL_CHO_01": {
        "dialogue": 1.0, "interaction": 2.0, "conflict": 5.0, "pressure": 5.0,
        "tension": 4.0, "relationship": 1.0, "goal": 3.5, "character": 3.0
    },
    "CAL_CHO_02": {
        "dialogue": 1.0, "interaction": 2.0, "conflict": 5.0, "pressure": 5.0,
        "tension": 4.5, "relationship": 4.0, "goal": 4.0, "character": 4.0
    },
    "CAL_REV_01": {
        "dialogue": 3.5, "interaction": 4.0, "conflict": 3.0, "pressure": 3.5,
        "tension": 5.0, "relationship": 5.0, "goal": 4.0, "character": 4.0
    },
    "CAL_REV_02": {
        "dialogue": 1.0, "interaction": 1.0, "conflict": 3.0, "pressure": 4.0,
        "tension": 5.0, "relationship": 2.5, "goal": 5.0, "character": 3.0
    },
}


def load_calibration_scenes() -> Dict[str, str]:
    """加载校准集场景文本（硬编码，便于独立运行）"""
    # 从之前的 JSON 中提取（此处简化，实际应从 data/calibration_set 读取）
    # 为简洁，这里用占位文本；实际运行时应从文件加载
    scenes = {}
    # 注意：实际验证时需要从 calibration_scenes.json 读取完整文本
    # 这里只做示例框架，实际文本请从 data/calibration_set 加载
    return scenes


def compute_metrics(engine: NarrativeKPIEngine, text: str, state_before: Dict, state_after: Dict) -> Dict[str, float]:
    """计算引擎输出"""
    result = engine.compute(text, state_before, state_after)
    return {
        "dialogue": result.dialogue,
        "interaction": result.interaction,
        "conflict": result.conflict,
        "pressure": result.pressure,
        "tension": result.tension,
        "relationship": result.relationship,
        "goal": result.goal,
        "character": result.character,
    }


def calculate_mae(predictions: Dict[str, Dict], ground_truth: Dict[str, Dict]) -> Dict[str, float]:
    """计算平均绝对误差（MAE）"""
    dims = ["dialogue", "interaction", "conflict", "pressure", "tension",
            "relationship", "goal", "character"]

    mae = {dim: 0.0 for dim in dims}
    count = 0

    for scene_id, pred in predictions.items():
        if scene_id not in ground_truth:
            continue
        gt = ground_truth[scene_id]
        for dim in dims:
            mae[dim] += abs(pred.get(dim, 0) - gt.get(dim, 0))
        count += 1

    for dim in dims:
        mae[dim] = mae[dim] / count if count > 0 else 0.0

    return mae


def main():
    """主验证流程"""
    print("=" * 60)
    print("KPI Engine 验证 (Phase 5 Gate 3)")
    print("=" * 60)

    engine = NarrativeKPIEngine()

    # 模拟：使用 CAL_EXP_01 作为示例
    # 实际验证需要完整文本和状态，这里打印框架
    print("\n✅ KPI Engine 已初始化")
    print(f"   角色列表: {engine.character_names}")
    print(f"   支持维度: {list(HUMAN_SCORES['CAL_EXP_01'].keys())}")

    print("\n📊 验证结果（模拟）:")
    print("   - 需要从 data/calibration_set 加载完整场景文本")
    print("   - 需要构造 state_before / state_after")
    print("   - 计算 MAE / Pearson 相关性")

    # 示例：计算 CAL_EXP_01 的简单测试
    test_text = "林逸推开沉重的石门，灰尘簌簌落下..."
    state_before = {"relationships": {}, "global_flags": {}, "characters": {}}
    state_after = {"relationships": {}, "global_flags": {}, "characters": {}}

    result = engine.compute(test_text, state_before, state_after)

    print(f"\n   测试场景 (CAL_EXP_01 模拟):")
    print(f"   - Narrative Value: {result.narrative_value:.2f}")
    print(f"   - Engagement: {result.engagement:.2f}, Progression: {result.progression:.2f}")
    print(f"   - Dialogue: {result.dialogue:.1f}, Character: {result.character:.1f}")

    print("\n" + "=" * 60)
    print("下一步：加载完整校准集，计算 MAE，验证 Gate 3 (r > 0.7)")


if __name__ == "__main__":
    main()