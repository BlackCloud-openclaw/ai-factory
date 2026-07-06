from next.kernel.world import KernelWorldState
from next.kernel.plugin import DomainPlugin
from next.kernel.entity import Entity, EntityType
from next.kernel.capability import Capability, CapabilityMode

class KernelFactory:
    @staticmethod
    def create(plugin: DomainPlugin) -> KernelWorldState:
        kernel = KernelWorldState()

        # 根据插件配置初始化一些默认实体（如主角）
        char_cfg = plugin.get_character_config()
        protagonist_name = char_cfg.get("protagonist", {}).get("default_name", "主角")
        protagonist_entity = Entity(
            id=protagonist_name,
            name=protagonist_name,
            type=EntityType.CHARACTER,
            attributes={
                "hp": char_cfg.get("protagonist", {}).get("initial_hp", 100),
                "energy": char_cfg.get("protagonist", {}).get("initial_energy", 100),
            }
        )
        kernel.entities[protagonist_name] = protagonist_entity

        # 初始化能力（等级）
        rank_config = plugin.get_rank_config()
        rank_capability = Capability(
            name="rank",
            mode=CapabilityMode.DISCRETE,
            value=char_cfg.get("protagonist", {}).get("initial_rank", rank_config["levels"][0]),
            metadata={"levels": rank_config["levels"]}
        )
        kernel.capabilities[f"{protagonist_name}|rank"] = rank_capability

        # 将世界规则存储到 kernel.metadata 中，供后续验证使用
        kernel.metadata["world_rules"] = plugin.get_world_rules()
        kernel.metadata["themes"] = plugin.get_themes()
        kernel.metadata["conflict_keywords"] = plugin.get_conflict_keywords()

        return kernel