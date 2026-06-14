import yaml
from pathlib import Path

_config = None

def _get_config():
    global _config
    if _config is None:
        cfg_path = Path("config/xianxia/character.yaml")
        if cfg_path.exists():
            with open(cfg_path) as f:
                _config = yaml.safe_load(f) or {}
        else:
            _config = {}
    return _config

def get_protagonist_id(world, compressed=None, *args):
    return _get_config().get("protagonist", {}).get("default_name", "林逸")

def get_protagonist_progression(world, compressed=None, *args):
    pid = get_protagonist_id(world)
    char = world.characters.get(pid)
    return char.full_realm() if char else "unknown"

def get_item_owner(world, compressed=None, item_name=None):
    if not item_name:
        return "none"
    for name, char in world.characters.items():
        if item_name in char.inventory:
            return name
    return "none"

def get_relationship_value(world, compressed=None, from_char=None, to_char=None):
    if not from_char or not to_char:
        return 0
    key = f"{from_char}|{to_char}"
    return world.relationships.get(key, 0)

def get_active_arcs_count(world, compressed=None, *args):
    if not compressed:
        return 0
    arcs = getattr(compressed, 'character_arcs', {}) or {}
    return sum(1 for status in arcs.values() if status != "resolved")

def get_major_conflict(world, compressed=None, *args):
    if world.global_flags.get("sect_civil_war"):
        return "宗门内乱"
    if world.global_flags.get("demon_cult_invasion"):
        return "魔道入侵"
    return "unknown"
