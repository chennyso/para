# 训练回执与预测验证

`verify_experiment_contracts.py` 读取只读的 `qwen30-schedule-20261108` 和 `qwen30-overnight-20261108` 归档。费用测试使用独立数据库，恢复验证复制原始 SQLite 数据库后进行。脚本核验逐 Rank 生效配置、微批分组、GPU 费用预留和登记、原生计算事件及 P2P 消息配对。记录中缺少异步请求的实际等待事件时，异步等待顺序保持未验证。

在 Mistral 的归档成本条件下，独立阶段模型可以评价 VPP 和层分区变化。TP 和微批大小变化缺少相容的计算与通信成本，PP 变化缺少相容的通信成本。相关候选保留 `NEEDS_PROFILE`。阶段费用是预测值；性能仍以正式训练回执为准。

## 测试环境和基准差异

使用 Python 3.12，在 macOS 上执行 CPU 侧测试，未运行新的 CUDA 或 NCCL 作业。对照代码 `ce60412` 与本次分支在下列五个测试文件中均有 73 项通过、9 项失败：`test_strategy_transformations.py`、`test_cost_model.py`、`test_physical_paths.py`、`test_global_planning.py`、`test_state_sharding_and_resource_replay.py`。

| 测试 | 记录的差异及覆盖影响 |
| --- | --- |
| `GlobalPlanningTests.test_30b_tensor_parallel_limit_applies_to_domains_and_explicit_intents` | 可选 TP 为 `[1,2,4,8]`，断言为 `[1,2]`；该用例未通过模型特定候选域检查。 |
| `GlobalPlanningTests.test_tensor_parallel_hard_limit_cannot_be_expanded` | 可选 TP 为 `[1,2,4,8]`，断言为 `[1,2,4]`；该用例未通过候选域上限检查。 |
| `GlobalPlanningTests.test_fsdp_v1_representative_keeps_source_checkpoint_format` | 目录中没有匹配的 FSDP 家族；该用例未覆盖 FSDP 检查点格式继承。 |
| `GlobalPlanningTests.test_global_family_catalog_covers_dense_and_expert_constraints` | 家族目录只包含 Megatron DDP；该用例未覆盖 FSDP 家族枚举。 |
| `StateShardingTests.test_dp_one_has_no_cross_rank_sharding_gain` | 适配器拒绝 ETP 大于一的 FSDP 设置；该用例未覆盖 DP1 所有权。 |
| `StateShardingTests.test_loading_and_training_peaks_are_separate` | 同一 ETP 限制；该用例未覆盖 FSDP 的加载和训练显存。 |
| `StateShardingTests.test_v1_accepts_existing_torch_dist_checkpoint` | 同一 ETP 限制；该用例未覆盖 FSDP 检查点接入。 |
| `StateShardingTests.test_v1_dense_zero_three_ownership` | 同一 ETP 限制；该用例未覆盖 FSDP 参数分片。 |
| `ResourceAwareReplayTests.test_differential_anchor_scope_excludes_global_family_change` | 同一 ETP 限制；该用例未覆盖 FSDP 条件下的预测锚点范围。 |
