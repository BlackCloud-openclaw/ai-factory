# src/writing/snapshot/writer.py

import os
import tempfile
from pathlib import Path
from typing import Optional

from src.writing.snapshot.models import PipelineSnapshot
from src.writing.snapshot.serializer import JsonSerializer, Serializer


class SnapshotWriter:
    def __init__(
        self,
        base_dir: Path,
        serializer: Optional[Serializer] = None,
    ):
        self.base_dir = base_dir
        self.serializer = serializer or JsonSerializer()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write(self, snapshot: PipelineSnapshot) -> Path:
        snapshot_id = snapshot.identity.snapshot_id
        final_path = self.base_dir / f"{snapshot_id}.snapshot"

        data = self.serializer.serialize(snapshot)

        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=self.base_dir,
                prefix=f"{snapshot_id}.tmp.",
                suffix=".snapshot",
            ) as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
                tmp_path = Path(f.name)

            tmp_path.replace(final_path)
            return final_path

        except Exception:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise