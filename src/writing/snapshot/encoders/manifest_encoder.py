# src/writing/snapshot/encoders/manifest_encoder.py

from typing import Any

from src.writing.snapshot.models import SnapshotManifest
from src.writing.snapshot.encoder_registry import Encoder, EncoderRegistry
from src.writing.snapshot.encoder_utils import encode_dataclass


class ManifestEncoder(Encoder):
    field_name = "manifest"

    def encode(self, value: Any) -> Any:
        if not isinstance(value, SnapshotManifest):
            raise TypeError(f"Expected SnapshotManifest, got {type(value).__name__}")
        return encode_dataclass(value)


EncoderRegistry.register("manifest", ManifestEncoder())