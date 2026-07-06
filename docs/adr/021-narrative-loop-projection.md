决策：Narrative Projection Layer 正式加入系统架构，作为独立架构层保留。

理由：

    能稳定生成并持久化

    可被 Planner 消费

    可量化监控（FPS/LPS/APS/QPS）

    可支撑 A/B 实验

范围：Projection Database 保留全部字段（Loop、Focus、Attention、Question），供离线分析、Dashboard、Debug、Research 使用。