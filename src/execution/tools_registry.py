"""Tools registry for managing agent tools."""

import importlib
import importlib.util
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.common.logging import setup_logging

logger = setup_logging("execution.tools_registry")


@dataclass
class ToolInfo:
    """Metadata for a registered tool."""

    name: str
    description: str
    module_path: str
    function_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolsRegistry:
    """Registry for managing agent tools.

    Tools are stored as Python modules in a tools directory.
    Each tool module must export a `get_tool_info()` function that returns
    a dict with at least: name, description, module_path, function_name.
    """

    def __init__(self, tools_dir: str = "/tmp/ai_factory/tools"):
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(parents=True, exist_ok=True)

        self._tools: Dict[str, ToolInfo] = {}
        self._tool_modules: Dict[str, Any] = {}

        self._scan_existing_tools()

    def _scan_existing_tools(self) -> None:
        """Scan tools_dir for .py files and register them."""
        if not self.tools_dir.exists():
            return

        for py_file in self.tools_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            module_name = py_file.stem
            try:
                spec = importlib.util.spec_from_file_location(
                    module_name, str(py_file)
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if hasattr(module, "get_tool_info"):
                        info = module.get_tool_info()
                        tool = ToolInfo(
                            name=info.get("name", module_name),
                            description=info.get("description", ""),
                            module_path=str(py_file),
                            function_name=info.get("function_name", ""),
                            parameters=info.get("parameters", {}),
                            metadata=info.get("metadata", {}),
                        )
                        self._tools[tool.name] = tool
                        self._tool_modules[tool.name] = module
                        logger.info(f"Auto-registered tool: {tool.name}")
            except Exception as e:
                logger.warning(f"Failed to load tool from {py_file}: {e}")

    def register_tool(
        self,
        name: str,
        description: str,
        module_path: str,
        function_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolInfo:
        """Register a tool.

        Args:
            name: Unique tool name
            description: Tool description
            module_path: Path to the Python module
            function_name: Name of the function to call
            parameters: Tool parameters schema
            metadata: Additional metadata

        Returns:
            ToolInfo for the registered tool
        """
        tool = ToolInfo(
            name=name,
            description=description,
            module_path=module_path,
            function_name=function_name,
            parameters=parameters or {},
            metadata=metadata or {},
        )

        self._tools[name] = tool
        logger.info(f"Registered tool: {name}")
        return tool

    def get_tool(self, name: str) -> Optional[ToolInfo]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            ToolInfo or None if not found
        """
        return self._tools.get(name)

    def get_tool_function(self, name: str) -> Optional[Callable]:
        """Get a loaded tool function by name.

        Args:
            name: Tool name

        Returns:
            Callable or None if not found
        """
        tool = self._tools.get(name)
        if not tool:
            return None

        if name not in self._tool_modules:
            module = self.load_tool_module(name)
            if module is None:
                return None

        module = self._tool_modules[name]
        func = getattr(module, tool.function_name, None)
        if func is None or not callable(func):
            logger.error(
                f"Tool {name}: function '{tool.function_name}' not found in module"
            )
            return None
        return func

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools.

        Returns:
            List of tool info dicts
        """
        tools = []
        for name, tool in self._tools.items():
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "module_path": tool.module_path,
                    "function_name": tool.function_name,
                    "parameters": tool.parameters,
                    "metadata": tool.metadata,
                }
            )
        return tools

    def load_tool_module(self, name: str) -> Optional[Any]:
        """Dynamically load a tool module.

        Args:
            name: Tool name

        Returns:
            Module object or None if loading failed
        """
        tool = self._tools.get(name)
        if not tool:
            logger.error(f"Tool '{name}' not found in registry")
            return None

        try:
            spec = importlib.util.spec_from_file_location(
                name, tool.module_path
            )
            if not spec or not spec.loader:
                logger.error(f"Cannot load spec for {tool.module_path}")
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._tool_modules[name] = module
            logger.info(f"Loaded tool module: {name}")
            return module

        except Exception as e:
            logger.error(f"Failed to load tool module '{name}': {e}")
            return None

    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the registry.

        Args:
            name: Tool name

        Returns:
            True if removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            self._tool_modules.pop(name, None)
            logger.info(f"Removed tool: {name}")
            return True
        return False

    def clear(self) -> None:
        """Clear all tools from the registry."""
        self._tools.clear()
        self._tool_modules.clear()
        logger.info("Cleared all tools from registry")

    def save_registry(self, path: Optional[str] = None) -> str:
        """Save registry metadata to a JSON file.

        Args:
            path: Output path (default: tools_dir/registry.json)

        Returns:
            Path to saved file
        """
        if path is None:
            path = str(self.tools_dir / "registry.json")

        registry_data = {
            "tools_dir": str(self.tools_dir),
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "module_path": t.module_path,
                    "function_name": t.function_name,
                    "parameters": t.parameters,
                    "metadata": t.metadata,
                }
                for t in self._tools.values()
            ],
        }

        with open(path, "w") as f:
            json.dump(registry_data, f, indent=2)

        logger.info(f"Registry saved to {path}")
        return path

    def load_registry(self, path: Optional[str] = None) -> int:
        """Load registry metadata from a JSON file.

        Args:
            path: Registry file path (default: tools_dir/registry.json)

        Returns:
            Number of tools loaded
        """
        if path is None:
            path = str(self.tools_dir / "registry.json")

        if not os.path.exists(path):
            logger.error(f"Registry file not found: {path}")
            return 0

        with open(path, "r") as f:
            data = json.load(f)

        count = 0
        for tool_data in data.get("tools", []):
            tool = ToolInfo(**tool_data)
            self._tools[tool.name] = tool
            count += 1

        logger.info(f"Loaded {count} tools from {path}")
        return count
