"""Simple sandbox using subprocess."""

import subprocess
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, Any, Union
from src.common.logging import setup_logging

logger = setup_logging("execution.sandbox")

class CodeSandbox:
    def __init__(self, timeout: int = 30, work_dir: str = "/tmp/ai_factory"):
        self.timeout = timeout
        self.work_dir = work_dir
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    
    async def execute(self, code_or_path: Union[str, Path], language: str = "python", **kwargs) -> Dict[str, Any]:
        """Execute code from string or file path."""
        logger.info(f"Executing {language} code from: {code_or_path[:50] if isinstance(code_or_path, str) else code_or_path}")
        
        if language != "python":
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Language {language} not supported",
                "returncode": -1
            }
        
        # Check if it's a file path
        if isinstance(code_or_path, (str, Path)) and Path(code_or_path).exists():
            file_path = str(Path(code_or_path))
        else:
            # It's code string, create temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=self.work_dir) as f:
                f.write(code_or_path)
                file_path = f.name
        
        python_exe = sys.executable
        
        try:
            result = subprocess.run(
                [python_exe, file_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.work_dir
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Timeout after {self.timeout}s",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
        finally:
            # Don't delete temp file if it was created
            pass
