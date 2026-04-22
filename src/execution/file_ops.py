"""File operations module for reading, writing, and managing files."""

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional


class FileOperations:
    """Safe file operations for the AI Factory."""

    def __init__(self, base_dir: str = "/tmp/ai_factory"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def read_file(self, file_path: str) -> str:
        """Read file content."""
        resolved = self._resolve_path(file_path)
        with open(resolved, "r", encoding="utf-8") as f:
            return f.read()

    def write_file(self, file_path: str, content: str) -> str:
        """Write content to a file."""
        resolved = self._resolve_path(file_path)
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        return resolved

    def append_file(self, file_path: str, content: str) -> None:
        """Append content to a file."""
        resolved = self._resolve_path(file_path)
        with open(resolved, "a", encoding="utf-8") as f:
            f.write(content)

    def delete_file(self, file_path: str) -> bool:
        """Delete a file."""
        resolved = self._resolve_path(file_path)
        if os.path.exists(resolved):
            os.remove(resolved)
            return True
        return False

    def list_files(
        self,
        directory: Optional[str] = None,
        pattern: str = "*",
        recursive: bool = False,
    ) -> list[str]:
        """List files matching a pattern."""
        base = self._resolve_path(directory or "")
        glob_pattern = f"**/{pattern}" if recursive else pattern
        found = []
        for path in Path(base).glob(glob_pattern):
            if path.is_file():
                found.append(str(path))
        return sorted(found)

    def copy_file(self, source: str, destination: str) -> str:
        """Copy a file."""
        src = self._resolve_path(source)
        dst = self._resolve_path(destination)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return dst

    def create_temp_file(
        self, content: str, suffix: str = ".py", prefix: str = "ai_factory_"
    ) -> str:
        """Create a temporary file with the given content."""
        temp_dir = os.path.join(self.base_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        file_id = uuid.uuid4().hex[:8]
        file_path = os.path.join(temp_dir, f"{prefix}{file_id}{suffix}")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path

    def get_file_info(self, file_path: str) -> dict[str, Any]:
        """Get file metadata."""
        resolved = self._resolve_path(file_path)
        stat = os.stat(resolved)
        return {
            "path": resolved,
            "name": os.path.basename(resolved),
            "size": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "is_file": os.path.isfile(resolved),
            "is_dir": os.path.isdir(resolved),
        }

    def _resolve_path(self, file_path: str) -> str:
        """Resolve and validate file path within base directory."""
        if not file_path:
            return self.base_dir

        resolved = os.path.abspath(os.path.join(self.base_dir, file_path))

        # Security check: ensure path is within base directory
        if not resolved.startswith(os.path.abspath(self.base_dir)):
            raise ValueError(
                f"Access denied: {file_path} is outside the allowed directory"
            )

        return resolved

    def ensure_directory(self, directory: str) -> str:
        """Ensure a directory exists within the base directory."""
        resolved = self._resolve_path(directory)
        os.makedirs(resolved, exist_ok=True)
        return resolved
