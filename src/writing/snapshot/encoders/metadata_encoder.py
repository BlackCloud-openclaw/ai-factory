# src/writing/snapshot/encoders/metadata_encoder.py

from typing import Any
from src.writing.snapshot.models import SnapshotMetadata
from src.writing.snapshot.encoder_registry import Encoder, EncoderRegistry


class MetadataEncoder(Encoder):
    field_name = "metadata"
    
    def encode(self, value: Any) -> Any:
        if not isinstance(value, SnapshotMetadata):
            return {}
        return {
            "runtime_version": value.runtime_version,
            "writer_version": value.writer_version,
            "llm_model": value.llm_model,
            "temperature": value.temperature,
            "seed": value.seed,
            "git_commit": value.git_commit,
            "git_dirty": value.git_dirty,
            "experiment_id": value.experiment_id,
            "python_version": value.python_version,
            "platform": value.platform,
            "os": value.os,
            "dependency_hash": value.dependency_hash,
        }


EncoderRegistry.register("metadata", MetadataEncoder())