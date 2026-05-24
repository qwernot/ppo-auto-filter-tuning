# reference outputs

本目录存放当前 `PPO` 滤波器主线已经冻结、可直接用于论文与学生接续的参考快照。

## 当前包含

- `campaigns/filter_example_one_local_bulk_320_r1`
  - 已完成的 320 组局部真实 Sonnet 仿真摘要
- `campaigns/filter_example_one_length_augmented_probe_v1`
  - 已冻结的长度增强探测计划快照，用于后续扩展，不代表已完成结果

## 约定

- 本目录内容作为“已发布参考资产”保留。
- 学生后续新增的仿真结果建议写入 `outputs/generated`。
- 论文正文优先引用本目录与 `models/reference_runs` 中的冻结结果。
