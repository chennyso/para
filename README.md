# ParaMeld

面向大语言模型训练的智能体驱动混合并行规划。这里提供 2026 年 11 月 2 日的论文、核心代码与主要实验数据。

## 论文

- [中文论文 PDF（正文、参考文献与附录）](paper/ParaMeld-Chinese.pdf)
- [LaTeX 源文件与矢量图片](paper/ParaMeld-Chinese-source.zip)

论文分别说明完整策略核验、条件分区、可选路径构造与训练反馈。逐轮试验图与Lightning容量修复表保留真实策略变化、结果选择及失败状态。

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
