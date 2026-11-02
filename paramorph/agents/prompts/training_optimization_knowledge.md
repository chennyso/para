# TrainingOptimizationKnowledge

所有硬性条件以当前源码和实测 source_hashes 为准。后端约束入口为 paramorph/validation/parallel_validator.py、paramorph/core_data_models.py 和 megatron/core/parallel_state.py；调度入口为 megatron/core/pipeline_parallel/schedules.py。成本记录保存软件来源摘要，跨版本复用需要重新验证。

TP 在同层内划分 Tensor，影响计算形状和 Collective。PP 将连续层映射到 Pipeline Rank。CP 划分序列，DP 划分样本，EP 分配 Expert。EP 不作为独立的 world_size 乘数；具体 Folding 和 ETP 条件由 ParallelValidator 判断。固定 GPU 数量要求 TP*PP*CP*DP 等于实际 world_size。

MoE需要共同检查本地专家数量、专家FFN分片宽度与Token分配。每个EP Rank持有num_experts/EP名专家，ETP进一步切分专家权重；模型结构中的moe_intermediate_size与ETP确定本地FFN宽度。CP、MBS与Token Dispatcher同时影响每名专家收到的Token数量，其实际分布需要路由记录。扩大CP或ETP可能增加小型Grouped GEMM的占比；减少通信与参数容量后，继续检查实际矩阵形状和算子成本。使用本次实测比较整项联合策略，分别保留总FLOPs、算子粒度和通信依赖的信息。

GBS=DP*MBS*num_microbatches。MBS 和梯度累积共同决定激活驻留、消息频率和 Pipeline Bubble。VPP 将多个连续 Region 放到同一物理 Rank，必须累计其参数、Optimizer 和同时驻留的激活。原生调度及通信依赖由 ProgramAgent 和 MicrobatchScheduler 生成。

Region Partition 改变连续层的实际 Placement，保持层顺序和覆盖完整性。计算时间均衡需要考虑 Attention、MLP、MoE、Mamba、GDN 的不同成本以及首尾 Embedding 和 Loss；层数相同无法保证耗时相同。

Tensor Sharding Layout 必须存在实际执行入口和兼容通信，独立 Wgrad 和任意通信重排需要后端支持。Recomputation 通过额外计算减少保存的激活，通信重叠需要真实 Stream、Collective 顺序及配对测量。相关编译入口为 paramorph/agents/program_agent.py 和 strategy_transformations.py。

显存包括参数、梯度、Optimizer、激活、通信 Buffer、Workspace 和 Runtime Overhead。检查物理 Rank 峰值及安全余量。预测数据来源为 paramorph/predictor/fast_predictor.py，精确执行条件匹配由 profile_database.py 执行；缺失成本必须请求 Profiling。

训练失败后先识别失败阶段及原始信息。NaN或Inf需要检查Loss、梯度、Loss Scaling、Optimizer更新与实际生效精度；恢复成功更新以后再评价吞吐。容量失败需要定位发生于权重加载、前向保存、反向计算还是Optimizer临时空间，并检查候选操作能否改变对应对象。继承已经成功的运行条件，明确下一项完整策略解决哪项失败或新增代价；相同失败的适用条件随后端源码及完整配置保存。

瓶颈诊断需要区分实测 CUDA Event 与模拟时间线。未经校准的跨 Rank 时钟不能证明通信等待。Anchor 加法校准调整时间尺度，候选排名需要独立测量验证。盲测时排除测试计划自身的 Anchor、显存校准和验证残差。
