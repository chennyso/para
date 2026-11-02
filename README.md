# ParaMeld

面向大语言模型训练的智能体驱动混合并行规划。此仓库提供 2026 年 11 月 2 日的论文、核心源码与经过隐私处理的实验数据。

## 论文

- [中文论文 PDF](paper/ParaMeld-Chinese.pdf)
- [中文补充材料 PDF](paper/ParaMeld-Chinese-Supplement.pdf)
- [LaTeX 与原始矢量图片](paper/ParaMeld-Chinese-source.zip)

## 源码

`paramorph/` 保留当前项目的原始 Python 包名称，包含四角色 Agent、混合并行策略生成、约束补全、条件分区、执行图、成本预测、实验记录和训练适配器。

| 目录或文件 | 内容 |
| --- | --- |
| `paramorph/agents/` | Agent 实现、提示词、策略变换与工具调用 |
| `paramorph/search/` | 全局策略调查、联合规划、条件分区与候选选择 |
| `paramorph/graph/` | 模型结构、执行图构建与执行诊断 |
| `paramorph/predictor/` | 计算、通信、状态分片与显存成本 |
| `paramorph/storage/` | 候选、测量和调用记录 |
| `paramorph/experiments/` | 搜索入口、预测核查与成本测量程序 |
| `integration/megatron/core/frontier/` | Megatron 模块、参数加载和并行执行扩展 |
| 根目录的 `*_train.py` | 模型与混合并行训练入口 |

搜索依赖安装：

```bash
python -m pip install -r paramorph/requirements.txt
```

先将 `integration/megatron/core/frontier/` 放入兼容 Megatron-LM 源码的对应目录，将该工作目录加入 `PYTHONPATH`。在具有实际模型、硬件和成本数据库配置的环境中运行 CPU 搜索：

```bash
export PYTHONPATH=/path/to/Megatron-LM:$PWD
python paramorph/experiments/run_system_search.py \
  --config /path/to/system-config.json \
  --output search-output \
  --method full --no-gpu
```

真实训练需要兼容的 Megatron-LM、PyTorch、CUDA、Transformer Engine、NCCL、模型权重及数据。训练集成代码按原有相对路径放入相应 Megatron-LM 工作目录。Agent 服务使用运行环境中的认证配置。模型权重、训练数据与认证文件由使用者自行配置。

## 实验数据

| 路径 | 内容 |
| --- | --- |
| `data/pvc/strategies.csv` | 历史 PVC 策略参数与测量索引 |
| `data/pvc/complete-measurements.csv` | 具有完整 Rank 计时记录的历史策略子集 |
| `data/agents/trials.csv` | 本次实验的逐轮生效配置、状态与测量结果 |
| `data/measurements/fp16-fp32-pairs.csv` | 四模型、八项完整 GPU 配对测量 |
| `data/measurements/per-rank/` | 配对测量的生效配置及各 Rank 原始数值字段 |
| `data/search/controlled-48-runs.csv` | 48 次 CPU 搜索的 DAG 预测与费用记录 |
| `data/optico/trials.csv` | OptiCo 作者实现的逐轮配置和实际结果 |
| `data/archival/` | 原论文主表、系统比较和组件消融的归档数值 |

本次四模型配对采用 FP16 计算，主参数、主梯度和 Adam 状态使用 FP32；每个策略独立运行一次，预热 5 次更新，测量 30 次更新。吞吐按全局每次更新的 Token 数量除以最慢 Rank 的平均更新时间计算；显存取所有 Rank 的最大分配量。MFU 使用各模型记录的每次更新 FLOPs 与每张 GPU 209.5 TFLOP/s 的参考峰值计算。

历史记录保留各自精度、批量、序列长度、设备数量及协议字段。不同训练条件的数据按记录分别使用。失败与未完成状态保留在逐轮记录中，缺失测量保持空值。CPU 搜索记录中的时间为 DAG 预测值。OptiCo 数据保留作者后端和提交身份，其后端与 ParaMeld 的训练后端存在差异。归档主表和消融的原始重复测量对应关系尚未全部恢复，归档状态为 `PARTIAL_RAW_SUPPORT`。

检查公开数据和文件摘要：

```bash
python scripts/verify_data.py
```

`MANIFEST.json` 列出所有公开文件的 SHA-256；`SOURCE_MANIFEST.json` 保留复制的源码与论文文件摘要。数据导出删除集群访问地址、私人绝对路径和认证信息，数值与状态保留原始记录。

## 版权

各源码文件保留原有版权声明。Megatron 集成部分的上游许可位于 [integration/LICENSE.Megatron-LM](integration/LICENSE.Megatron-LM)。
