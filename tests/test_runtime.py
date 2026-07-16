# tests/test_runtime.py
import pytest
from src.runtime import StateRouter, SceneFeatures, PredictionCapability, RealizationCapability, RetryStrategy

def test_router_highly_open():
    router = StateRouter()
    features = SceneFeatures(transition_rigidity=0.33)
    cap = router.route(features)
    assert cap.prediction == PredictionCapability.PRIMARY
    assert cap.realization == RealizationCapability.ENHANCED
    assert cap.retry == RetryStrategy.FULL

def test_router_competitive():
    router = StateRouter()
    features = SceneFeatures(transition_rigidity=0.55)
    cap = router.route(features)
    assert cap.prediction == PredictionCapability.ASSIST
    assert cap.realization == RealizationCapability.ENHANCED

def test_router_moderately_rigid():
    router = StateRouter()
    features = SceneFeatures(transition_rigidity=0.75)
    cap = router.route(features)
    assert cap.prediction == PredictionCapability.DISABLED
    assert cap.realization == RealizationCapability.NORMAL

def test_router_rigid():
    router = StateRouter()
    features = SceneFeatures(transition_rigidity=0.92)
    cap = router.route(features)
    assert cap.prediction == PredictionCapability.DISABLED
    assert cap.realization == RealizationCapability.NONE