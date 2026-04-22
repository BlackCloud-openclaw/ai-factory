"""Document ingestion module for parsing and chunking documents."""

import os
import uuid
import hashlib
from typing import Any, Optional
from pathlib import Path

from src.config import config
from src.common.logging import setup_logging

logger = setup_logging("knowledge.ingestion")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".py"}


class DocumentIngestor:
    """Parse documents (.txt, .md, .py) into chunks for vector storage."""

    def __init__(
        self,
        chunk_size: int = config.chunk_size,
        chunk_overlap: int = config.chunk_overlap,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_file(self, file_path: str) -> list[dict[str, Any]]:
        """Ingest a single file and return chunks."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}"
            )

        content = self._read_file(file_path)
        metadata = {
            "source_path": str(file_path),
            "file_name": path.name,
            "file_type": ext.lstrip("."),
            "file_size": path.stat().st_size,
        }

        chunks = self._chunk(content, metadata)
        logger.info(f"Ingested {path.name}: {len(chunks)} chunks created")
        return chunks

    def ingest_directory(
        self, directory: str, recursive: bool = True
    ) -> list[dict[str, Any]]:
        """Ingest all supported files from a directory."""
        dir_path = Path(directory)
        all_chunks = []

        pattern = "**/*" if recursive else "*"
        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    chunks = self.ingest_file(str(file_path))
                    all_chunks.extend(chunks)
                except (FileNotFoundError, ValueError) as e:
                    logger.warning(f"Skipping {file_path}: {e}")

        logger.info(
            f"Ingested {len(all_chunks)} total chunks from {directory}"
        )
        return all_chunks

    def _read_file(self, file_path: str) -> str:
        """Read file content as text."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _chunk(
        self, content: str, metadata: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Split content into overlapping chunks."""
        chunks = []
        chars = list(content)
        i = 0

        while i < len(chars):
            chunk_text = "".join(chars[i : i + self.chunk_size])

            # Try to break at word boundaries
            if i + self.chunk_size < len(chars):
                last_space = chunk_text.rfind(" ")
                if last_space > self.chunk_size * 0.5:
                    chunk_text = chunk_text[: last_space + 1].strip()

            if not chunk_text:
                break

            chunk_id = str(uuid.uuid4())
            chunk_index = len(chunks)

            # Compute content hash for deduplication
            content_hash = hashlib.md5(chunk_text.encode()).hexdigest()

            chunks.append(
                {
                    "id": chunk_id,
                    "content": chunk_text,
                    "chunk_index": chunk_index,
                    "content_hash": content_hash,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunk_id": chunk_id,
                    },
                }
            )

            i += self.chunk_size - self.chunk_overlap

        return chunks
