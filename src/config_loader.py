from pathlib import Path
import yaml

class XianxiaConfig:
    def __init__(self, base_dir: str = "config/xianxia"):
        self.base_dir = Path(base_dir)
        self.character = self._load("character.yaml")
        self.rank = self._load("rank.yaml")
        self.entropy = self._load("entropy.yaml")
        self.perception = self._load("perception.yaml")
        self.voice = self._load("voice.yaml")
        self.theme = self._load("theme.yaml")
        self.projection = self._load("projection.yaml")
        self.cognitive_rules = self._load("cognitive_rules.yaml")  # 新增

    def _load(self, name: str):
        path = self.base_dir / name
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {}

_config = None

def get_xianxia_config():
    global _config
    if _config is None:
        _config = XianxiaConfig()
    return _config
