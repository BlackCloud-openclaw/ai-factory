markdown

# Domain Identity 架构备注

## 设计原则

### 1. ID 是系统标识，Name 是显示名称

所有业务逻辑使用 **ID** 进行操作：

```python
protagonist_id = get_main_character_id()
char = world_state.characters.get(protagonist_id)

仅在最终输出时使用 Name：
python

display_name = get_character_name(protagonist_id)

2. 角色识别基于 Tags

主角通过 tags: [main] 识别，而非硬编码 id: protagonist。
yaml

characters:
  - id: captain       # 科幻题材可以是 captain
    name: 凯尔markdown

# Domain Identity 架构备注

## 设计原则

### 1. ID 是系统标识，Name 是显示名称

所有业务逻辑使用 **ID** 进行操作：

```python
protagonist_id = get_main_character_id()
char = world_state.characters.get(protagonist_id)

仅在最终输出时使用 Name：
python

display_name = get_character_name(protagonist_id)

2. 角色识别基于 Tags

主角通过 tags: [main] 识别，而非硬编码 id: protagonist。
yaml

characters:
  - id: captain       # 科幻题材可以是 captain
    name: 凯尔
    tags: [main]
  - id: detective     # 侦探题材可以是 detective
    name: 福尔摩斯
    tags: [main]

3. 别名支持实体消歧
yaml

artifacts:
  - id: jade_pendant
    name: 神秘玉佩
    aliases:
      - 古玉
      - 玉佩
      - 灵玉

resolve_artifact("古玉") 返回 "jade_pendant"。
架构约束（强制）
禁止

    ❌ 业务代码中直接使用角色名字符串：
    python

    world_state.characters.get("林逸")   # 禁止
    if "林逸" in text:                   # 禁止

    ❌ 在业务层传播 get_main_character_name() 作为业务键：
    python

    protagonist_name = get_main_character_name()
    world_state.characters.get(protagonist_name)  # 禁止

允许

    ✅ 使用 ID 作为业务键：
    python

    protagonist_id = get_main_character_id()
    world_state.characters.get(protagonist_id)

    ✅ 仅在显示时使用名称：
    python

    display_name = get_character_name(protagonist_id)
    logger.info(f"主角 {display_name} 正在修炼")

    ✅ Prompt 模板中使用变量：
    python

    prompt = f"你扮演的是 {get_main_character_name()} 的修仙故事"

检查命令
bash

# 检查业务代码中是否还有硬编码角色名（不应输出业务代码）
grep -rnE '["\x27](林逸|韩立|苏清月|玄老|二叔)["\x27]' src/ --include="*.py" \
  | grep -v "config/" \
  | grep -v "domain/" \
  | grep -v "tests/" \
  | grep -v "docs/"

热加载支持

load_character_config() 使用全局缓存，调用 reload_character_config() 可清除缓存：
python

from src.domain.identity import reload_character_config

# 配置文件变更后
reload_character_config()

后续集成文件监控即可实现热加载。

    tags: [main]
  - id: detective     # 侦探题材可以是 detective
    name: 福尔摩斯
    tags: [main]

3. 别名支持实体消歧
yaml

artifacts:
  - id: jade_pendant
    name: 神秘玉佩
    aliases:
      - 古玉
      - 玉佩
      - 灵玉

resolve_artifact("古玉") 返回 "jade_pendant"。
架构约束（强制）
禁止

❌ 业务代码中直接使用角色名字符串：
python

world_state.characters.get("林逸")   # 禁止
if "林逸" in text:                   # 禁止

❌ 在业务层传播 get_main_character_name() 作为业务键：
python

protagonist_name = get_main_character_name()
world_state.characters.get(protagonist_name)  # 禁止

允许

✅ 使用 ID 作为业务键：
python

protagonist_id = get_main_character_id()
world_state.characters.get(protagonist_id)

✅ 仅在显示时使用名称：
python

display_name = get_character_name(protagonist_id)
logger.info(f"主角 {display_name} 正在修炼")

✅ Prompt 模板中使用变量：
python

prompt = f"你扮演的是 {get_main_character_name()} 的修仙故事"

检查命令
bash

# 检查业务代码中是否还有硬编码角色名（不应输出业务代码）
grep -rnE '["'"'"'](林逸|韩立|苏清月|玄老|二叔)["'"'"']' src/ --include="*.py" \
  | grep -v "config/" \
  | grep -v "domain/" \
  | grep -v "tests/" \
  | grep -v "docs/"

    注意：上述命令中的 "'"'"' 是为了在 shell 中正确表示单引号。如果遇到问题，可以简化为：
    bash

    grep -rnE '"(林逸|韩立|苏清月|玄老|二叔)"' src/ --include="*.py" | grep -v config | grep -v domain | grep -v tests

热加载支持

load_character_config() 使用全局缓存，调用 reload_character_config() 可清除缓存：
python

from src.domain.identity import reload_character_config

# 配置文件变更后
reload_character_config()

后续集成文件监控即可实现热加载。
迁移状态
项目	状态
配置抽离 (character.yaml)	✅ Phase 1
ID 优先接口 (identity.py)	✅ Phase 1
业务代码清理 (src/*.py)	🔄 Phase 1 进行中
WorldState key 迁移 (name -> id)	📅 Phase 2
Prompt 模板变量化	📅 Phase 2
Benchmark 参数化	📅 Phase 3
认知规则配置化	📅 Phase 4
Domain Plugin	📅 Phase 5
text


---

## 总结

| 项目 | 评价 |
|------|------|
| **内容准确性** | ✅ 完全正确，符合架构设计 |
| **格式规范性** | ⚠️ 需要修正代码块标记和语言标签后的换行 |
| **可执行性** | ✅ 命令和示例均可执行 |
| **完整性** | ✅ 覆盖了设计原则、约束、检查命令、热加载和迁移状态 |

**建议**：将文档内容替换为上面修正后的版本，确保在 Markdown 渲染器中正常显示。内容本身不需要改动。

