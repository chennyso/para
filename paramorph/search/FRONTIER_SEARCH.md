# 增量定价与完整执行评价

`FrontierPricer` 按物理 Stage 逐个扩展其全部 VPP 区间。每个部分状态记录已分配区间、固定端点域和执行家族身份。`PartitionDomain.complete_fixed()` 使用端点可达性动态规划，提前删除无法连接成完整连续分区的状态。

每次 LP 更新后，待展开状态使用当前对偶价格重新排序。完整候选通过原有 `PhysicalStageOracle.column()` 构造真实候选列，再交给 `NativeEventMaster`；最终分区仍然执行完整 Projection 和显存评价。

## 排序模型的下界

固定原生事件模板后，已有 `estimated_column()` 的约化成本可以写成：

\[
R(I_0,\ldots,I_{V-1})=R_{seed}+\sum_{v=0}^{V-1}\Delta_v(I_v).
\]

各项变化包含该虚拟区间的前向与反向任务费用、按参数状态字节数缩放的 Optimizer 和状态通信费用，以及端点一致性约束的对偶价格。固定通信接口的对偶贡献包含在种子常数中。

`LinearPricingBound` 对尚未分配的各个区间分别取最小变化，放松它们之间的连接条件：

\[
L(P)=R_{seed}+\sum_{v\in P}\Delta_v(I_v)
       +\sum_{v\notin P}\min_{I_v}\Delta_v(I_v).
\]

任何合法完整续接都属于这些独立区间域的笛卡尔积，因此该数值不超过相应续接的排序模型费用。完整状态的下界等于该状态的排序模型费用。

该结论的范围限定为已有线性定价排序模型。真实 Projection 还包含共享资源竞争、通信和显存生命周期，当前实现保持 `pricing_certified=false`，通过下界安排访问顺序。状态展开数量不足时保留待展开状态，并明确记录搜索尚未完成。

## 执行边界和候选组合

`ExecutionNeighborhood` 比较父配置与候选的完整执行图，记录任务变化、相邻依赖、Collective 参与者、全部共享资源接口，以及受影响物理 Rank 的 Buffer 生命周期和参数状态。共享资源记录覆盖当前观察图中的全部相关任务；显存约束保留相关 Rank 的全部生命周期记录。

当前时间关系属于对应完整图的观察证据。结果明确记录 `independence_certified=false` 和 `requires_complete_replay=true`。

Oracle 要求固定主问题内的任务身份、Rank、任务类型、依赖和资源集合相同。改变这些结构时需要建立新的主问题。

跨物理 Stage 的通信实例携带参与 Stage 和完整接口签名。整数主问题为每个接口值增加一致性等式，要求参与 Stage 选择相同接口。LP 约化成本包含这些等式对应的真实对偶价格。

## 成本身份和 Agent 程序

`FragmentCostCache` 的身份包含完整请求、成本版本、Profile 身份、种子执行图和模块费用。缓存复用相同身份的原子模块费用求和，完整执行时间、通信竞争和显存准入仍由完整评价器计算。

`PricingPolicy` 包含阶段优先顺序、区间扩展规则、显存与通信排序权重、每次 CPU 搜索的状态展开数量，以及真实证据引用。`FrontierPricer(generate=...)` 在当前结构内调用一次 Agent，随后多轮 LP 求解复用同一程序。结构或成本身份改变后，原定价器拒绝继续执行。

`PortfolioControl.execute_physical_stage_program()` 接收实际 Provider 的 `generate` 接口，执行上述搜索，并保存程序身份及调用次数。`JointPlanning.solve_physical_stage_partition()` 保留原有入口，支持显式传入增量定价器。

实验入口 `run_physical_stage_comparison.py` 支持 `frontier` 和 `agent_frontier`。前者使用明确的确定性扩展策略，后者使用真实 Agent Provider。搜索和整数主问题默认没有运行时限。

## 验证范围

`verify_frontier_pricer.py` 使用归档 MiMo 原生运行成本，检查完整候选集合、排序下界、首批候选费用、重复调用状态和整数事件 Replay。可通过 `--agent-transport` 调用当前真实 Agent 中转，验证一次调用产生程序、连续两批复用程序。

`verify_execution_interfaces.py` 使用两个归档完整预测，验证跨 Stage 通信接口等式、整数目标和独立枚举一致性，并检查实际 Buffer 生命周期接口。

`verify_physical_stage_master.py` 保留归档成本、整数主问题、小规模枚举及完整 Projection 的验证。缺少成本时继续输出 `NEEDS_PROFILE`。

## 跨前缀状态支配

`ExecutionBoundaryIndex` 从同一成本版本的完整原生DAG评价和逐卡容量准入记录建立续接证书。固定执行家族中，两个前缀需要具有相同深度、相同末端位置、相同合法后缀集合、相同设备容量及预测实现版本。每个合法后缀都必须存在完整评价，证书才成立。

默认目标为在逐卡容量约束下最小化完整步时。若前缀A在每个合法后缀上的完整步时均不高于前缀B，且至少一项严格降低，A支配B。任何B的合法完整方案都能够替换为同后缀的A方案，保持逐卡容量可行并且完整步时不增加，因此删除B保留声明域内的最优目标值。`time_and_memory`目标还要求逐卡峰值逐项不增加。

该证书覆盖整个后缀域的完整Replay结果，包含共享资源竞争和FSDP参数生命周期的最终影响。已认证状态的资源签名使用公共边界身份，完整前缀继续保存为配置见证。缺少后缀、完整费用、预测版本或逐卡容量证据时，使用原有前缀身份并保留状态。不同模型、执行家族和成本版本之间不能传递支配结论。

## 离线程序学习与在线调用

`train_pricing_programs.py`读取真实归档运行及原生DAG，通过scikit-optimize的Gaussian Process贝叶斯优化，调整显存权重、通信权重、状态展开数量、阶段顺序及区间扩展规则。标签来源区分完整DAG评价和归档复合事件定价模型，保存每次实际CPU搜索的原始结果。

Random Forest根据执行图特征学习程序参数与排序类别。模型清单登记训练执行图、费用版本、原始Profile、实际搜索记录、特征字段、软件版本和文件校验值。独立执行图验证禁止重复使用训练图。

`planning_pricing_program_path`向在线Strategy Agent提供已训练程序先验。Agent根据当前执行证据决定具体程序；同一程序在内部LP更新中复用，连续收益不足时重新生成。`planning_partition_engine=frontier`将该流程接入正式四角色控制器；TP、PP、VPP、DP、CP、FSDP及重计算的家族选择继续由上层完整联合搜索决定。

## 请求归属与恢复

`PVCStructuredProvider`和`pvc_transport`使用实际认证API、原生JSON Schema及无请求时限的独立内存会话。请求登记Pod、控制器PID和进程创建时间，中转按提交时间选择仍有实际控制器接收的请求。控制器退出后的请求保留原文并登记取消状态，其他Pod的请求保持原状。控制器迁移保留历史测量、费用和已完成轮次。

解释字符串的长度约束与既有提示一致，执行参数域保持原生合法范围。Agent程序不设置GPU运行时限；没有完整DAG预测和逐卡显存准入的候选不能启动正式训练。
