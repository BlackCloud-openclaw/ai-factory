# src/runtime/scene_analyzer.py
"""
Scene Analyzer - 场景分析器
"""

import logging
from typing import Dict, Any, Optional, List

from src.runtime.models import SceneAnalysis, AnalysisSource

logger = logging.getLogger(__name__)


class SceneAnalyzer:
    def __init__(self, measured_db: Optional[Dict[str, float]] = None):
        self.measured_db = measured_db or self._default_measured_db()
        logger.info(f"SceneAnalyzer initialized with {len(self.measured_db)} measured scenes")

    def _default_measured_db(self) -> Dict[str, float]:
        return {
            "scene_reunion": 0.33,
            "scene_dilemma": 0.71,
            "scene_meeting": 0.86,
            "scene_letter": 1.00,
            "mp1": 0.50,
            "mp2": 0.60,
            "mp3": 0.90,
        }

    def analyze(self, scene_plan: Dict[str, Any]) -> SceneAnalysis:
        scene_id = scene_plan.get("scene_id", "unknown")
        if scene_id in self.measured_db:
            tr = self.measured_db[scene_id]
            return SceneAnalysis(
                tr=tr,
                prediction_plasticity=1.0 - tr,
                source=AnalysisSource.MEASURED,
                confidence=1.0,
                state_type=scene_plan.get("state_type", ""),
                features=scene_plan.get("scene_features", {}),
                reason=f"scene_id '{scene_id}' 已在 Phase 5 实验中被验证"
            )
        features = scene_plan.get("scene_features", {})
        if features:
            tr, conf = self._infer_from_features(features)
            return SceneAnalysis(
                tr=tr,
                prediction_plasticity=1.0 - tr,
                source=AnalysisSource.INFERRED,
                confidence=conf,
                state_type=scene_plan.get("state_type", ""),
                features=features,
                reason=f"从 scene_features 推断 (confidence={conf:.2f})"
            )
        return SceneAnalysis(
            tr=0.50,
            prediction_plasticity=0.50,
            source=AnalysisSource.DEFAULT,
            confidence=0.30,
            state_type=scene_plan.get("state_type", ""),
            features={},
            reason="无 scene_features，使用默认值 0.50"
        )

    def _infer_from_features(self, features: Dict) -> tuple:
        obs = features.get("observations", {})
        judgements = features.get("judgements", {})
        script_strength = judgements.get("script_strength", self._infer_script_strength(obs))
        choice_competition = judgements.get("choice_competition", self._infer_choice_competition(obs))
        raw_tr = 0.5 + 0.3 * script_strength - 0.2 * choice_competition
        num_chars = obs.get("num_active_characters", 1)
        raw_tr -= 0.03 * (num_chars - 1)
        if obs.get("explicit_deadline", False):
            raw_tr += 0.05
        alt_actions = obs.get("alternative_actions", 1)
        raw_tr -= 0.02 * (alt_actions - 1)
        tr = max(0.05, min(0.95, raw_tr))
        judgement_keys = ["script_strength", "choice_competition"]
        present = sum(1 for k in judgement_keys if k in judgements)
        confidence = 0.5 + 0.25 * (present / len(judgement_keys))
        if obs:
            confidence = min(0.9, confidence + 0.1)
        return tr, confidence

    def _infer_script_strength(self, obs: Dict) -> float:
        if obs.get("explicit_deadline", False):
            return 0.7
        if obs.get("num_active_characters", 1) <= 1:
            return 0.65
        return 0.5

    def _infer_choice_competition(self, obs: Dict) -> float:
        alt = obs.get("alternative_actions", 1)
        if alt >= 4:
            return 0.8
        elif alt >= 3:
            return 0.6
        elif alt >= 2:
            return 0.4
        else:
            return 0.2

    def add_measurement(self, scene_id: str, tr: float):
        self.measured_db[scene_id] = tr
        logger.info(f"Added measurement: {scene_id} = {tr:.3f}")


def analyze_scene(scene_plan: Dict[str, Any]) -> SceneAnalysis:
    return SceneAnalyzer().analyze(scene_plan)


def analyze_scenes(scene_plans: List[Dict[str, Any]]) -> List[SceneAnalysis]:
    analyzer = SceneAnalyzer()
    return [analyzer.analyze(sp) for sp in scene_plans]