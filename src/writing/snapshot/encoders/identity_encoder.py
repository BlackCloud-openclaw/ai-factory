# src/writing/snapshot/encoders/identity_encoder.py

from typing import Any

from src.writing.snapshot.models import SnapshotIdentity
from src.writing.snapshot.encoder_registry import Encoder, EncoderRegistry


class IdentityEncoder(Encoder):
    field_name = "identity"

    def encode(self, value: Any) -> Any:
        if not isinstance(value, SnapshotIdentity):
            raise TypeError(f"Expected SnapshotIdentity, got {type(value).__name__}")
        return {
            "snapshot_id": str(value.snapshot_id),
        }


EncoderRegistry.register("identity", IdentityEncoder())