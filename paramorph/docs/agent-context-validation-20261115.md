Agent 的联合策略 Schema、原生参数范围和工具查询能力保持完整。请求使用 JSON Reference 共享相同记录；还原后的观测与原观测逐字段相等。

测量阶段直接记录程序计算的预测编号、成本状态、逐卡容量与允许操作。策略生成后的语义审核保留，测量选择继续由 ExperimentInvestigationAgent 决定。结构核验按父配置、候选实际配置、后端来源、合法参数范围、GPU 数量和约束实现版本复用。每批补测及正式执行前重新检查当前成本版本的完整 DAG、同负载最佳正式实测和全部 GPU 的显存准入。

使用 Qwen2.5-7B、16384 上下文第一轮的原始认证请求，记录状态固定，执行新的真实 Agent 调用：

| 指标 | 原记录 | 修改后的真实调用 |
|---|---:|---:|
| Agent 请求次数 | 13 | 9 |
| API 输入 tokens | 420364 | 315578 |
| 认证 API 调用累计秒数 | 549.144599 | 357.016023 |

输入减少24.9%，认证 API 调用累计耗时减少35.0%。原轮次的 Provider 等待记账为588.688996秒；表格统一使用认证回执的 API 调用计时。GPU 校准及完整轮次墙钟耗时分别计数。

13份原始观测均通过逐字段还原验证。九份新响应通过原严格 Schema。新 JointStrategyAgent 响应中的两项完整联合策略通过原生形状核验及 GraphActionCompiler 编译。

第一轮五项真实候选重新执行准入检查，均保持未超过11521.342105 tokens/s领先正式实测的判定。实际容量失败记录继续排除。原生结构核验首次耗时0.920001秒，复用耗时0.473549秒；该项单独记录。

PVC 请求传输采用 gzip、Base64、原始字节数和 SHA-256 校验。认证回执、请求归属、原始请求及响应均保留。部署入口源文件为 experiments/pvc_agent_transport_entry.py，复制到实验根目录的 agent_transport.py 后执行。

真实请求、响应、计时和验证记录位于实验目录 verification/context-reuse-authenticated-20261115，以及原 Pod 的 verification/context-reuse-20261115/native-review-round1-records。全部验证未启动 GPU 运行，运行时限保持未设置。
