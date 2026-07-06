import yaml
from pathlib import Path
from next.kernel.plugin import DomainPlugin

class SciFiPlugin(DomainPlugin):
    @property
    def domain_name(self) -> str:
        return "scifi"

    def _load_yaml(self, filename: str) -> dict:
        path = Path(f"config/scifi/{filename}")
        if not path.exists():
            raise FileNotFoundError(f"Config file {path} not found")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def get_rank_config(self) -> dict:
        return self._load_yaml("rank.yaml")

    def get_world_rules(self) -> list:
        data = self._load_yaml("world_rules.yaml")
        return data.get("rules", [])

    def get_themes(self) -> list:
        data = self._load_yaml("theme.yaml")
        return data.get("themes", [])

    def get_conflict_keywords(self) -> list:
        data = self._load_yaml("theme.yaml")
        return data.get("conflict_keywords", [])

    def get_character_config(self) -> dict:
        return self._load_yaml("character.yaml")