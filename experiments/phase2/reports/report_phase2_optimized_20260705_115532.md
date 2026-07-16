# Phase 2 实验报告（Incremental Writer 优化版）

生成时间: 2026-07-05 11:55:32

## 各组成绩

| 组 | 执行方式 | 分段数 | Surface | Constraint | Outcome | Overall |
|---|---|---|---|---|---|---|
| D | single | 1 | 0.175 | 0.000 | 0.083 | 0.086 |
| I2 | incremental | 2 | 0.550 | 0.100 | 0.267 | 0.306 |
| I3 | incremental | 3 | 0.608 | 0.200 | 0.225 | 0.344 |
| I4 | incremental | 4 | 0.500 | 0.444 | 0.194 | 0.380 |

## 结论

**最佳总体 Control Score**: 组 I4 (0.380)

### 对比

- 单次执行平均 Overall: 0.086
- 增量执行平均 Overall: 0.343
- **增量执行提升: +0.257**