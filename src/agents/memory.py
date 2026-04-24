"""Memory Agent - Provides context memory for AI Factory workflow."""

import time
import uuid
from typing import Any, Optional

from src.common.logging import setup_logging

logger = setup_logging("agents.memory")


class MemoryAgent:
    """In-memory storage agent for maintaining conversation and code generation history.

    Memories are organized by project_id to isolate different projects.
    Each memory entry has a key-value pair with optional metadata.
    """

    def __init__(self):
        self.memories: dict[str, dict[str, dict[str, Any]]] = {}
        logger.info("MemoryAgent initialized")

    async def store(self, project_id: str, key: str, value: Any, metadata: dict = None) -> str:
        """Store a memory entry for a given project.

        Args:
            project_id: Project identifier to group memories.
            key: Memory key (e.g., 'last_code', 'user_preferences').
            value: The memory value to store.
            metadata: Optional metadata (timestamp, source, etc.).

        Returns:
            The unique memory_id for this entry.
        """
        memory_id = uuid.uuid4().hex[:12]

        if project_id not in self.memories:
            self.memories[project_id] = {}

        entry = {
            "id": memory_id,
            "value": value,
            "metadata": metadata or {},
        }

        if "timestamp" not in entry["metadata"]:
            entry["metadata"]["timestamp"] = time.time()

        self.memories[project_id][key] = entry

        logger.debug(f"Stored memory: project={project_id}, key={key}, id={memory_id}")
        return memory_id

    async def retrieve(self, project_id: str, key: str) -> Any:
        """Retrieve a memory value by project_id and key.

        Args:
            project_id: Project identifier.
            key: Memory key to retrieve.

        Returns:
            The stored value, or None if not found.
        """
        if project_id not in self.memories:
            return None

        entry = self.memories[project_id].get(key)
        if entry is None:
            logger.debug(f"Memory not found: project={project_id}, key={key}")
            return None

        logger.debug(f"Retrieved memory: project={project_id}, key={key}")
        return entry["value"]

    async def retrieve_with_metadata(self, project_id: str, key: str) -> dict[str, Any] | None:
        """Retrieve a memory entry including metadata.

        Args:
            project_id: Project identifier.
            key: Memory key to retrieve.

        Returns:
            Full entry dict with id, value, and metadata, or None if not found.
        """
        if project_id not in self.memories:
            return None

        entry = self.memories[project_id].get(key)
        if entry is None:
            return None

        return entry

    async def list_all(self, project_id: str) -> dict[str, Any]:
        """List all memories for a given project.

        Args:
            project_id: Project identifier.

        Returns:
            Dict mapping keys to their values for this project.
        """
        if project_id not in self.memories:
            return {}

        result = {}
        for key, entry in self.memories[project_id].items():
            result[key] = entry["value"]

        logger.debug(f"Listed {len(result)} memories for project={project_id}")
        return result

    async def update(self, project_id: str, key: str, value: Any) -> bool:
        """Update an existing memory entry.

        Args:
            project_id: Project identifier.
            key: Memory key to update.
            value: New value to store.

        Returns:
            True if updated, False if key didn't exist.
        """
        if project_id not in self.memories:
            return False

        if key not in self.memories[project_id]:
            logger.debug(f"Cannot update non-existent memory: project={project_id}, key={key}")
            return False

        entry = self.memories[project_id][key]
        entry["value"] = value
        entry["metadata"]["updated_at"] = time.time()

        logger.debug(f"Updated memory: project={project_id}, key={key}")
        return True

    async def delete(self, project_id: str, key: str) -> bool:
        """Delete a memory entry.

        Args:
            project_id: Project identifier.
            key: Memory key to delete.

        Returns:
            True if deleted, False if key didn't exist.
        """
        if project_id not in self.memories:
            return False

        if key not in self.memories[project_id]:
            return False

        del self.memories[project_id][key]
        logger.debug(f"Deleted memory: project={project_id}, key={key}")
        return True

    async def delete_project(self, project_id: str) -> int:
        """Delete all memories for a project.

        Args:
            project_id: Project identifier.

        Returns:
            Number of memories deleted.
        """
        if project_id not in self.memories:
            return 0

        count = len(self.memories[project_id])
        del self.memories[project_id]
        logger.info(f"Deleted all {count} memories for project={project_id}")
        return count

    async def get_project_keys(self, project_id: str) -> list[str]:
        """Get all memory keys for a project.

        Args:
            project_id: Project identifier.

        Returns:
            List of memory keys.
        """
        if project_id not in self.memories:
            return []
        return list(self.memories[project_id].keys())

    async def append_to_memory(self, project_id: str, key: str, value: Any, max_items: int = 50) -> None:
        """Append a value to an existing list memory, keeping only the last max_items.

        Useful for conversation history or code generation log.

        Args:
            project_id: Project identifier.
            key: Memory key (should point to a list).
            value: Value to append.
            max_items: Maximum number of items to keep.
        """
        existing = await self.retrieve(project_id, key)

        if existing is None:
            existing = []
        elif not isinstance(existing, list):
            existing = [existing]

        existing.append(value)

        # Keep only the last max_items
        if len(existing) > max_items:
            existing = existing[-max_items:]

        await self.store(project_id, key, existing)
        logger.debug(f"Appended to list memory: project={project_id}, key={key}, total={len(existing)}")
