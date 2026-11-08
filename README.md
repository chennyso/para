# ParaMeld

资源感知训练依赖图引导的多智能体混合并行优化。这里提供 2026 年 11 月 2 日的论文、核心代码与主要实验数据。

## 论文

- [中文论文 PDF（正文、参考文献与附录）](paper/ParaMeld-Chinese.pdf)
- [LaTeX 源文件与矢量图片](paper/ParaMeld-Chinese-source.zip)

论文结合资源感知训练依赖图、瓶颈关联子图、反事实执行分析与多智能体联合规划，说明原生核验、条件分区、路径求解和成本敏感性如何参与策略搜索。正文、用户指定的40篇参考文献与六项附录共同编译在一份PDF中，参考文献使用GB/T 7714—2025顺序编码格式，发表时间截至2026年8月1日。逐轮试验图与Lightning容量修复表保留真实策略变化、结果选择及失败状态。

## 代码

`paramorph/` 保留项目原有 Python 包名称，包含四角色 Agent、策略生成、约束补全、条件分区、执行图、成本预测、实验记录与训练适配器。

| 路径 | 内容 |
| --- | --- |
| `paramorph/agents/` | Agent、提示词、工具调用与联合策略变换 |
| `paramorph/search/` | 全维度策略搜索、条件分区与候选保留 |
| `paramorph/graph/` | 模型结构、执行图与执行诊断 |
| `paramorph/predictor/` | 计算、通信、显存及分片成本 |
| `paramorph/experiments/` | 正常搜索、性能预测与测量入口 |
| `integration/megatron/core/frontier/` | Megatron 训练集成代码 |
| 根目录的 `*_train.py` | 训练执行入口 |

安装搜索依赖：

```bash
python -m pip install -r paramorph/requirements.txt
```

训练需要兼容的 Megatron-LM、PyTorch、CUDA、Transformer Engine 与 NCCL 环境。将 `integration/megatron/core/frontier/` 放入 Megatron-LM 的相应目录，配置实际模型、硬件、成本数据库与 Agent 服务认证后，可以调用搜索入口：

```bash
export PYTHONPATH=/path/to/Megatron-LM:$PWD
python paramorph/experiments/run_system_search.py \
  --config /path/to/system-config.json \
  --output search-output --method full --no-gpu
```

有图和无图复测使用两份独立配置，分别设置 `agent_ablation` 为
`full` 和 `without_dag_evidence`。两份配置共用模型、硬件、初始实测、
后端合法性检查及预算，使用不同的输出目录。设置
`selection_policy: "investigation"`、`single_decision_round: true` 和
`budgeted_investigation_enabled: true`，可以由 Agent 比较所有调查提出的
候选并选择下一项实验。逐轮回执记录完整生效配置、选择依据、预测不确定性和
实际 GPU·小时。GPU 锁等待时间单独记录。Megatron 训练回执保存实际执行的
微批调度表和 P2P 配置；缺少匹配的成本证据时，预计收益保留为空。

正式消融使用各自独立的搜索目录：

| `agent_ablation` | 运行设置 |
| --- | --- |
| `full` | Agent、原生事件图预测和条件联合搜索 |
| `without_dag_model` | 阶段计算与通信成本、流水线气泡估计；屏蔽图反馈 |
| `without_agent` | 确定性候选生成和选择；使用相同的事件图和训练后端 |
| `without_joint_search` | 保留 Agent 和 DAG；配置为单次决策和预算选择，候选只允许一个独立调节轴 |

`without_dag_evidence` 用于只考察图反馈的影响，其成本预测仍使用事件图。
阶段估计仅复用满足执行条件的实测阶段、链路与优化器成本；缺失条件保持待测。
各组应共享训练协议、硬件与 GPU·小时上限。独立重复使用不同的输出目录及种子，
完整数值等价仅在专门验证通过的回执中声明。
五模型的共同预算、测量窗口和重复种子见
[`five_model_protocol.json`](paramorph/experiments/five_model_protocol.json)。

以下验证脚本读取历史 MiMo、Falcon 和 FSDP 回执，以及新共同基线的逐 Rank
调度记录：

```bash
export PYTHONPATH="$PWD/paramorph:$PWD/integration:$PYTHONPATH"
python paramorph/tests/verify_budgeted_schedule_receipts.py \
  --archive /path/to/overnight-results \
  --native-run /path/to/qwen14/common/training/runs/run-id
python paramorph/tests/verify_experiment_contracts.py \
  --campaign /path/to/five-model-results \
  --archive /path/to/overnight-results \
  --output local-work/verification.sqlite
```

`paramorph/experiments/evaluate_prediction_pairs.py` 接受 JSONL 盲测配对，
每行包含 `prediction`、`measurement`、`measured_plan_signature` 和
`workload_fingerprint`，计算步时、显存及候选排序误差；预测缺失单独计数。

## 主要数据

| 路径 | 内容 |
| --- | --- |
| `data/archival/nine-model-summary.csv` | 原论文九模型主表，27 项数据 |
| `data/archival/component-ablation.csv` | 原论文组件消融，保留原值 |
| `data/archival/system-comparison.csv` | 原论文系统比较，保留原值 |
| `data/measurements/fp16-fp32-pairs.csv` | 四模型、八项完整 GPU 配对测量汇总 |
| `data/agents/trials.csv` | 本次实验 240 条候选的并行参数、生效策略和实际状态 |
| `data/optico/trials.csv` | OptiCo 作者实现的 18 次策略与结果 |
| `data/archival/nine-model-pvc-audit.csv` | 九模型主表的 PVC 来源与提升核查 |

四模型配对使用 FP16 计算，主参数、主梯度和 Adam 状态使用 FP32；每项策略独立运行一次，预热 5 次更新、测量 30 次更新。吞吐按最慢 Rank 的平均更新时间计算，显存使用所有 Rank 中的最大分配量。MFU 使用记录的模型 FLOPs 和每张 GPU 209.5 TFLOP/s 的参考峰值计算。这八项汇总已与 128 份逐 Rank 测量核对。

归档表格沿用原论文数值。重新读取700项历史运行和8项近期运行、复算10,640份逐Rank计时后，原主表27项吞吐与显存尚未找到同时匹配的记录，也未找到对应的三次重复组合，其数据来源状态保留为未核实。四模型近期配对测量与各自原始记录一致。失败和未完成候选保留原有状态，缺失测量保持空值。OptiCo数据保存作者后端及提交身份。完整历史PVC记录与逐Rank输出保存在实验档案中。

公开数据删除私人路径、集群访问地址和认证信息。`MANIFEST.json` 保存公开文件的 SHA-256，`SOURCE_MANIFEST.json` 保存源码与论文文件的原始摘要。

## 版权

源文件保留原有版权声明，Megatron 集成部分的上游许可见 [integration/LICENSE.Megatron-LM](integration/LICENSE.Megatron-LM)。
