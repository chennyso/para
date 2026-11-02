# ParaMorph 执行接口与证据要求

## 实现接口

| 接口 | 输入 | 输出 |
|---|---|---|
| `read_model` / `build_operator_graph` | 实际模型配置与工作负载 | 模块类型、Tensor Shape、依赖与算子语义 |
| `BottleneckGraph` | 后端完整执行 DAG、真实 Task ID | 分页子图、依赖、通信参与者、资源和反事实 Replay |
| `StrategyTransformationCompiler` | 完整计划、目标任务和动作 | 策略影响、物理资源与 GraphDelta |
| `ConditionalPartitionSolver.solve` | 固定 H、父计划、固定值、保留字段、允许范围和 CPU 预算 | 多份完整 L/Z 请求、分区证据、成本缺失和求解状态 |
| `CompleteAndRewrite` | 联合意图、后端域与完整候选 | 合法 TrainingPlan、执行 DAG、进程组与约束冲突 |
| `FamilyLedger` | H 身份、求解结果、候选与成本版本 | 家族状态、最佳预测、最佳实测和作用范围明确的证明 |
| `KernelCostSurface.query` | GPU、Kernel/Module、阶段、Shape、dtype、Layout | 实测、插值、具有依据的解析成本或 Profiling 请求 |
| `TopologyAwareNCCLCostSurface.query` | Collective、消息、物理组、拓扑、dtype、并发 | 实测曲线、物理链路成本或 Profiling 请求 |
| `FastPredictor.evaluate` | 完整请求、执行 DAG、成本模式 | Step Time、Tokens/s、逐 Rank 显存、关键路径和成本覆盖 |
| `replay_shared_cost_scenarios` | 多份完整预测、共享 `cost_key`、低高成本与资源模型 | 全部关联候选的情景 Replay 和三类排序变化 |
| `EvidenceGuidedSearch` | SystemConfig、成本数据库、Trial Ledger | 四 Agent 搜索、预算记录与最佳真实计划 |
| `MegatronAdapter` | 合法原生训练请求 | g5/g6 实际训练、Profiling、日志及失败原因 |

## Agent 与约束职责

完整流程使用 Bottleneck Analysis、Joint Strategy Generation、Constraint and Validation、Profiling and Evaluation 四个 Agent。约束 Agent 调用确定性求解工具并引用返回证据；计划的代数关系、完整层覆盖、Rank Mapping、PP/VPP 调度与后端能力由程序检查。

完整 Executable DAG、历史和成本库存放在后端。Agent 使用对应角色的紧凑上下文，按稳定 Task、Region、Flow 和证据 ID 分页查询。Token 审计区分 Provider 请求、工具调用、输入、输出、工具定义和工具返回。

策略查询阶段允许`intents=[]`，最终生成阶段检查配置的`intent_count`。完整策略明确提交全部执行取值，原生编译保持这些取值。变换预览与成对交互查询使用各自结构证据，引用经过类型核验；结构证据不提供未测量的性能收益。

`candidate_ids`使用已登记的`trial_id`，`inspect_global_families`通过`family_ids`或分页游标查询全局配置。已取得的全局取值和状态设置保存在`queried_global_families`，完整结果继续保留在后端。查询页面限制返回大小并给出`next_offset`，参数或页面约束冲突返回明确的约束反馈。

没有新增GPU测量的轮次也执行反馈，记录编译结果、预测状态、成本缺项及下一项调查。下一轮上下文包含近期预测候选与已完成轮次的反馈。Agent原始选择和协调器执行选择分别保存；关闭GPU执行时不自动补入新的GPU探索操作。总GPU预算允许为零。

## 家族与求解范围

家族身份仅包含 `TP/PP/CP/DP/EP/ETP/EDP/RankMapping`。VPP、MBS、Region、重计算、状态分片和执行后端属于同一 H 下的补全变量。Optimizer 数值精度及模型、GBS、序列长度、训练精度、GPU 数量固定。

连续模块 DP 使用明确边界和已分配资源签名。严格支配只在可比较状态之间执行；预算删除单独记录。`search_complete` 表示声明求解域的枚举状态，不能替代真实训练验证或完整目标最优性证明。逐层计算负载仅供分区排序，不直接作为完整 DAG 预测的严格下界。

`FEASIBLE` 需要完整计划的可行见证，预测见证与真实测量分别标注。`UNKNOWN` 保留没有见证或证明的家族。`INFEASIBLE` 仅接受覆盖全部家族补全的代数或穷尽证明。

一次完整计划 OOM 只说明该计划失败。只有该失败能够导出所有 L/Z 补全都必须违反的约束，并保存完整适用范围的证明时，才可以排除整个 H。固定意图、允许范围内无解、均匀分区失败、成本缺失和预算截断均不能自行扩大到全家族结论。

## 成本与排序证据

成本复用保留硬件、软件、源码、Kernel/Module、Shape、精度和执行条件。Module-composite 实测保持原有粒度；独立 Kernel 成本需要对应 Kernel 身份。改变全局并行度、物理映射和进程组后，重新构造相关通信、依赖与执行调度。

完整布局接续在每个分区深度集中执行评价。前缀按其端点到相容完整布局前缀端点的累计距离安排次序，同距使用局部负载；不同前缀轮流获得回放机会，后缀起点交替选择。完成本深度评价以后，`score`、`witness`与`retain`仅读取已有结果，不产生额外评价或费用。报告保存实际评价前缀、后缀来源，以及局部选择和最终保留状态各自的完整接续。完整接续评价一个完整候选，不提供该前缀所有完成方式的性能下界。

完整执行图包含模型实际启用的 Output、Loss、重复 MTP、梯度最终同步和 Optimizer 事件。显存按状态所有权、Buffer 分配释放及初始化、加载、训练、更新阶段计算。缺少必要成本保留 `NEEDS_PROFILE`，缺少显存条件保留 `REQUIRE_MEMORY_PROFILE`。

共享成本情景通过稳定 `cost_key` 同时修改全部关联候选，重新计算完整 DAG。排序检查包含最佳对象、Top-K 内部次序和 Top-K 成员。情景值与统计置信区间采用不同证据标签。成本版本变化使旧预测排序与相关下界失效，真实测量历史保留。

## 实验与预算

设备数量使用实际拓扑，当前可执行范围为 g5/g6 的八卡或十六卡。三十二卡属于任务书实验目标，需要对应硬件与真实运行记录后才能报告结果。

CPU 求解、完整预测、反事实 Replay、Agent 调用、Profiling GPU Hours 和完整训练分别计量。已有有效 Baseline 和成本记录按模型、工作负载、环境及代码身份复用。候选的真实 OOM、执行失败和性能结果进入 Trial Ledger。

CPU 验收使用真实模型配置、拓扑和历史成本检查合法补全、枚举一致性、预算状态及完整图依赖。端到端吞吐、独立显存与训练数值正确性分别需要真实 GPU 证据。当前能力声明不包含全局最优保证或三十二卡实测收益。
