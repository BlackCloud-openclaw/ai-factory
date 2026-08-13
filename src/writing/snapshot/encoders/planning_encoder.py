# src/writing/snapshot/encoders/planning_encoder.py

from typing import Any
from src.writing.artifact.planning import PlanningArtifact, PlanningCore, WorldStateArtifact
from src.writing.snapshot.encoder_registry import Encoder, EncoderRegistry


class PlanningEncoder(Encoder):
    field_name = "planning"
    
    def encode(self, value: Any) -> Any:
        if not isinstance(value, PlanningArtifact):
            return {}
        return {
            "schema_version": value.schema_version,
            "core": self._encode_core(value.core),
            "extension": dict(value.extension),
        }
    
    def _encode_core(self, core: PlanningCore) -> dict:
        return {
            "scene_id": core.scene_id,
            "scene_goal": core.scene_goal,
            "must_events": list(core.must_events),
            "world_state": self._encode_world_state(core.world_state),
            "conflicts": [self._encode_conflict(c) for c in core.conflicts],
            "characters": [self._encode_character(c) for c in core.characters],
            "emotion_arc": dict(core.emotion_arc) if core.emotion_arc else None,
        }
    
    def _encode_world_state(self, ws: WorldStateArtifact) -> dict:
        return {
            "location": ws.location,
            "time": ws.time,
            "weather": ws.weather,
            "realm": ws.realm,
        }
    
    def _encode_conflict(self, c) -> dict:
        return {
            "type": c.type,
            "description": c.description,
            "participants": list(c.participants),
            "severity": c.severity,
        }
    
    def _encode_character(self, c) -> dict:
        return {
            "id": c.id,
            "name": c.name,
            "role": c.role,
            "realm": c.realm,
        }


EncoderRegistry.register("planning", PlanningEncoder())