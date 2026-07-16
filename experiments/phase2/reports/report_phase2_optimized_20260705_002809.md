# Phase 2 实验报告（Incremental Writer 优化版）

生成时间: 2026-07-05 00:28:09

## 各组成绩

| 组 | 执行方式 | 分段数 | Surface | Constraint | Outcome | Overall |
|---|---|---|---|---|---|---|
| D | single | 1 | 0.348 | 0.300 | 0.183 | 0.277 |
| I2 | incremental | 2 | 0.620 | 0.300 | 0.258 | 0.393 |

## 结论

**最佳总体 Control Score**: 组 I2 (0.393)

### 对比

- 单次执行平均 Overall: 0.277
- 增量执行平均 Overall: 0.393
- **增量执行提升: +0.116**