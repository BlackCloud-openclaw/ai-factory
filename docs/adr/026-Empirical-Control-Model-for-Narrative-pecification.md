1. 标题

ADR-026: Empirical Control Model for Narrative Specification
2. 状态

Accepted
3. 背景

v2.0 已实现 Planning Contract（Promise）、Controlled Execution（Execution）、Validator（Verification）三层闭环。但 Planner 对 读者体验 的控制力缺失——Writer 仍需自行决定环境渲染、情绪轨迹、话语结构和视角锚定。

v2.1 的研究问题是：

    哪些结构化语义变量能够稳定、独立、可重复地控制 LLM 的叙事生成行为？

我们通过 Phase 4 实验（Counterfactual + Ablation + Isolation + Conflict）验证了四个候选假设。
4. 决策

基于 Phase 4 实验数据，我们确认以下 Empirical Control Model：
4.1 有效控制维度
维度	控制对象	Effect Size	稳定性
World	环境渲染（地点/时间/氛围/感官）	0.88	高
Emotion	读者情绪轨迹（begin/middle/end）	0.78	中-高
Function	话语结构（悬念/揭示/升级/释放/过渡）	0.72	最高
POV	信息可见性（视角锚定）	0.62	中

所有四个维度均验证有效，纳入 Specification。
4.2 优先级矩阵（Runtime 裁决规则）
text

POV > Function > World > Emotion

当冲突发生时，Runtime 按此优先级裁决。
4.3 边界条件

    World：在场景复杂度极高时可能失效（如集市+墓园混合）

    Emotion：在复杂场景中效力下降

    POV：最脆弱的维度，需额外验证

5. 工程映射
5.1 Specification Schema（冻结）
yaml

scene_spec:
  world:
    location: string       # 地点名称
    time: string          # 清晨/正午/黄昏/子夜/深夜
    atmosphere: string    # 氛围关键词
    sensory: [string]     # 2-4个感官细节
  reader_emotion:
    begin: string         # 开头情绪
    middle: string        # 中间情绪
    end: string           # 结尾情绪
  narrative_function: string  # introduce_mystery | escalate | reveal_truth | release_tension | transition | foreshadow
  pov: string            # 视角角色名

5.2 Planner 集成

Planner 在生成 PlanningContract 时必须同时生成 scene_spec 字段。
5.3 Writer 集成

Writer（ControlledWriter）必须读取 scene_spec 并将其转化为渲染指令。
5.4 Validator 集成

Validator 必须验证 scene_spec 是否被正确渲染（L1 表面检查 + L2 语义检查）。
5.5 Runtime 集成

Runtime 必须应用 Priority Matrix 裁决冲突。
6. 后果

正面：

    Writer 从"创作者"降级为"渲染器"

    读者体验首次被 Planner 控制

    实验验证了四个维度的因果效力

负面：

    场景复杂度可能使 World 失效

    需要额外稳定性优化（强制 JSON Schema）

7. 关联 ADR

    ADR-023: Planning Contract as Stable Interface

    ADR-024: Incremental Execution as Core Capability

    ADR-025: Validator as Controller

8. 参考

    Phase 4 实验报告：experiments/phase4/reports/summary.json

    样本数据：experiments/phase4/reports/raw/