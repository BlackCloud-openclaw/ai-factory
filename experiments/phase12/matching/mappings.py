from src.writing.events import EventType

DEFAULT_KEYWORD_MAPPING = {
    "获得": EventType.ITEM_ACQUIRE,
    "失去": EventType.ITEM_LOSE,
    "突破": EventType.REALM_UPGRADE,
    "进入": EventType.LOCATION_ENTER,
    "对话": EventType.DIALOGUE,
    "发现": EventType.DISCOVERY,
    "战斗": EventType.COMBAT_RESULT,
    "触发": EventType.PLOT_FLAG_SET,
}