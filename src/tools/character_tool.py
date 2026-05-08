# src/tools/character_tool.py
from src.writing.events import Event, EVENT_CHARACTER_UPDATE

def get_tool_info():
    return {
        "name": "update_character",
        "description": "更新角色状态（境界、生命值等）",
        "module_path": "",
        "function_name": "update_character",
        "parameters": {
            "name": {"type": "string", "description": "角色名"},
            "updates": {"type": "object", "description": "要更新的字段"}
        }
    }

def update_character(name: str, updates: dict, novel_id: str, chapter_id: str = None):
    """返回一个角色更新事件（不直接修改状态）"""
    return Event.new(
        EVENT_CHARACTER_UPDATE,
        {"name": name, "updates": updates},
        novel_id,
        chapter_id
    )