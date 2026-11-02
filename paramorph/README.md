# ParaMorph

ParaMorph: Multi-Agent-Guided Joint Hybrid Parallel Strategy Synthesis for Large Model Training。

ParaMorph 面向 Megatron-LM 生成完整训练计划，联合搜索全局混合并行、连续模块分区、执行布局和状态分片。实际 GPU 数量读取硬件拓扑；现有设备范围为 g5、g6，每个节点八张 GPU，可执行单节点八卡或双节点十六卡任务。任务书中的三十二卡属于实验目标，当前记录不构成三十二卡实测结果。

## 完整训练计划

训练计划表示为 `Theta=(H,L,Z)`。

- `H=(TP,PP,CP,DP,EP,ETP,EDP,RankMapping)`：全局并行结构与物理设备映射。
- `L=(VPP,MBS,Region,Recompute,Schedule)`：执行布局，包含合法 Microbatch 分组与通信重叠配置。
- `Z`：Megatron DDP 或已声明支持的 Megatron-FSDP 后端、参数与梯度分片、Optimizer 状态分片及 DeviceMesh。

一个全局家族只由 H 标识，同一 H 下的不同 L/Z 保存为完整候选。模型、GBS、序列长度、训练精度、Optimizer 定义与数值精度、GPU 数量在一次搜索中固定。Dense 设备关系为 `TP*PP*CP*DP=W`；专家侧使用实际 `ETP*EP*EDP*PP=W` 约束。CPU Optimizer Offload 不属于搜索空间。

## 四个 Agent

| 角色 | 代码身份 | 职责 |
|---|---|---|
| Bottleneck Analysis Agent | `PerformanceAnalysisAgent` | 根据真实测量、完整 DAG Replay 和历史记录查询瓶颈子图 |
| Joint Strategy Generation Agent | `JointStrategyAgent` | 提出新的全局家族或跨多个变量的联合意图 |
| Constraint and Validation Agent | `ConstraintValidationAgent` | 调用确定性补全、条件化分区与完整计划检查工具，引用实际求解结果 |
| Profiling and Evaluation Agent | `ExperimentInvestigationAgent` | 根据共享成本、排序变化和预算选择 Profiling 或真实训练 |

Agent 共享 TrainingPlan、Executable DAG、GraphDelta、Evidence、成本版本、测量记录和家族状态的身份引用。完整 DAG 保存在后端，工具分页返回真实 Task、Region、Flow、通信参与 Rank、依赖与证据。每个角色接收对应职责的紧凑上下文。完整四角色流程使用 `max_llm_calls_per_round=4`；工具追问的 Provider 请求独立计入总调用与 Token 预算。

## 条件化联合搜索

`ConditionalPartitionSolver` 固定 H，枚举声明支持的 VPP、MBS、重计算、Microbatch 分组、状态分片与相关执行选项。`fixed_values`、`preserve` 和 `allowed_ranges` 在候选枚举前共同约束搜索空间。

连续模块分区使用多候选动态规划 `D[p,j,sigma]`。状态保留模块覆盖位置、PP/VPP 归属、张量边界、已分配区间和资源身份。严格支配只允许在边界与资源生命周期签名相同时进行。宽度、扩展次数、布局数量或输出数量造成的删除分别记录预算终止原因，并设置 `search_complete=false`。

逐层实测成本用于分区负载排序，复用检查 GPU 型号、CUDA、Torch、代码来源、计算形状、精度和执行后端。成本缺失时记录 `NEEDS_PROFILE`，保留合法候选供完整预测和必要测量使用。分区负载分数只描述局部代理目标；候选 Step Time 由完整执行 DAG Replay 计算。当前求解器没有证明适用于完整 Replay 和全部 L/Z 的非零性能下界，下界字段保持未提供。

求解器输出完整请求与分区证据，`CompleteAndRewrite` 继续生成 TrainingPlan、Executable DAG、GraphDelta 和进程组。家族目录持续维护已有候选、当前成本版本下的最佳预测、最佳实测、待测成本和求解记录。

家族状态采用以下定义：

| 状态 | 证据要求 |
|---|---|
| `FEASIBLE` | 至少一份完整计划通过明确的逐 Rank 显存模型检查，或获得有效真实训练结果；记录预测与实测来源 |
| `UNKNOWN` | 尚未获得可行见证，也没有覆盖全家族的不可行证明 |
| `INFEASIBLE` | 固定 H 违反所有补全都必须满足的代数约束，或完整声明域已被确定性约束穷尽并证明不可行 |

单个计划 OOM、均匀分区失败、某个固定 L/Z 子域无解、成本缺失或预算耗尽，都不足以证明整个 H 不可行。一个失败记录只有同时给出与 L/Z 无关、覆盖全部允许补全的约束证明，才能支持家族排除。部分搜索没有全局最优性保证。

## 预测与预算化实验

Fast Predictor 按实际 GPU、Kernel 或 Module、阶段、Shape、dtype 与 Layout 查询计算成本，按 Collective、消息、Rank Group、物理路径和并发条件查询通信成本。相同共享成本具有稳定 `cost_key`，可以服务多份候选。

完整 Step Time 使用 Megatron 原生 PP/VPP 顺序、依赖和物理 Rank 资源模拟。执行图包含 Output、Loss、MTP、梯度最终同步与 Optimizer 任务。显存模型计算状态所有权、Activation 和通信 Buffer 生命周期、Workspace、安全余量，以及初始化、加载、训练和 Optimizer 阶段。

瓶颈调查可以将真实子图及其通信参与者、相关资源展开，并通过缩放指定任务成本重新 Replay，返回预测反事实影响。该值保留预测证据身份。

排序敏感性对同一 `cost_key` 同时更新所有关联候选的任务成本，重新运行完整资源 Replay，检查最佳候选变化、Top-K 内部逆序和成员变化。低、高成本值属于敏感性情景；只有具备独立统计依据时才能解释为置信区间。成本数据库更新后，相关预测和家族记录按版本重新评价。

Profiling 和 Benchmark 共用 GPU Hours 预算。最终输出来自通过真实训练且满足性能统计要求的候选。独立显存、完整梯度和 Optimizer 正确性记录分别保存，CPU 验收不等同于 GPU 训练通过。

## 执行与验证

配置结构位于 `contracts.py:SystemConfig`。四角色搜索入口：

```bash
python paramorph/experiments/run_system_search.py --config system-config.json --output search-output --method full
```

`--cost-mode` 支持 `hybrid`、`measured_only`、`analytical_only`。`--hardware-parameters` 读取具有来源的硬件与显存参数。`--no-gpu` 保存候选、预测与待执行请求，不启动 GPU 测量。外部 Agent 请求需要已有授权范围，四角色调用记录分别统计输入、输出、工具 Schema、工具返回与累计上下文 Token。

条件化分区的 CPU 验收使用已有模型、拓扑和真实 Profile：

```bash
python paramorph/experiments/validate_conditional_partition.py --config existing-system-config.json --output conditional-validation.json
```

验收检查连续分区与可穷举空间的一致性、VPP 归属、完整图依赖、预算状态、成本缺失、固定意图和不同后端覆盖。完整搜索实验仍需记录相同训练语义、实际设备、相同预算及独立测量。

## 核心代码

| 代码 | 职责 |
|---|---|
| `graph/model_graph.py` | 真实模型配置、混合模块类型和静态图 |
| `graph/bottleneck_analysis.py` | 真实子图展开、任务成本反事实与共享成本排序 Replay |
| `agents/strategy_transformations.py` | 策略影响、联合变换和 GraphDelta |
| `graph/complete_rewrite.py` | 原生约束补全与完整执行图编译 |
| `search/conditional_partition.py` | 固定 H 的多候选 L/Z 分区求解 |
| `search/joint_planning.py` | 全局家族、条件化求解、图预测与成本版本联动 |
| `search/family_state.py` | H 家族状态、候选与实测历史 |
| `predictor/fast_predictor.py` | 统一成本、显存与完整候选评价 |
| `predictor/compute_cost.py` / `predictor/communication_cost.py` | 计算与拓扑通信成本查询 |
| `predictor/dag_projection.py` | 原生执行图成本与动态显存预测 |
| `search/system_search.py` | 四角色、工具、预算、Profiling 与训练协调 |
| `storage/experiment_store.py` | Trial Ledger、证据和调用审计 |
| `adapters/megatron_adapter.py` | 实际设备范围与真实训练执行 |
