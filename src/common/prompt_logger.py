"""Prompt 快照日志 - 用于调试长程状态漂移"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

PROMPT_LOG_DIR = Path("logs/prompts")
PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)

ROLE_DIRS = ["writer", "planner", "validator"]
for d in ROLE_DIRS:
    (PROMPT_LOG_DIR / d).mkdir(exist_ok=True)


def compute_hash(obj: Any) -> str:
    """计算对象的确定性哈希"""
    from src.common.canonical import canonical_dumps
    return hashlib.sha256(canonical_dumps(obj).encode()).hexdigest()[:16]


def log_prompt(
    role: str,
    prompt: str,
    metadata: Optional[Dict[str, Any]] = None,
    response: Optional[str] = None,
    constraints: Optional[Dict[str, Any]] = None,
) -> str:
    """保存 prompt 快照，包含 constraint_hash"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    role_dir = PROMPT_LOG_DIR / role
    filename = role_dir / f"{timestamp}.txt"
    
    if metadata is None:
        metadata = {}
    
    # 添加 constraint_hash
    if constraints:
        metadata["constraint_hash"] = compute_hash(constraints)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        
        if metadata:
            f.write("=== METADATA ===\n")
            f.write(json.dumps(metadata, ensure_ascii=False, indent=2))
            f.write("\n\n")
        
        if constraints:
            f.write("=== CONSTRAINTS ===\n")
            f.write(json.dumps(constraints, ensure_ascii=False, indent=2))
            f.write("\n\n")
        
        f.write("=== PROMPT ===\n")
        f.write(prompt)
        f.write("\n\n")
        
        if response:
            f.write("=== RESPONSE ===\n")
            f.write(response[:8000])
            if len(response) > 8000:
                f.write(f"\n... (truncated, total {len(response)} chars)")
            f.write("\n")
    
    # 清理旧文件（保留最近 200 个）
    all_files = sorted(role_dir.glob("*.txt"))
    for old_file in all_files[:-200]:
        old_file.unlink()
    
    return str(filename)