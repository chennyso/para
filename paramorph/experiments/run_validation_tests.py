import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core_data_models import ModelSpec, HardwareTopology, ParallelConfig, LayerType, Candidate
from validation.parallel_validator import validate_candidate
from region_partition.region_partitioner import create_regions_for_candidate
from agents.architect_agent import ArchitectAgent
from history.history_manager import HistoryManager


def run_complete_validation_test():
    """完整的候选验证测试"""
    print("=" * 80)
    print("ParaMorph 候选验证测试")
    print("=" * 80)

    # 创建模型和硬件规格
    model_spec = ModelSpec(
        name="Lightning-30B",
        num_layers=52,
        hidden_size=6144,
        num_attention_heads=48,
        layer_types=[LayerType.ATTENTION] * 52,
        layer_order=list(range(52)),
    )

    hardware = HardwareTopology(
        num_gpus=16,
        num_nodes=2,
        gpus_per_node=8,
        memory_per_gpu_gb=80.0,
    )

    # 生成所有候选
    architect = ArchitectAgent(
        num_gpus=16,
        global_batch_size=16,
        model_num_layers=52,
        seq_length=4096
    )

    candidates = architect.generate_candidates_from_baseline(
        baseline_tp=2,
        baseline_pp=8,
        baseline_vpp=1
    )

    # 验证每个候选
    results = []
    for candidate in candidates:
        config = ParallelConfig(
            tp=candidate.tp,
            pp=candidate.pp,
            dp=candidate.dp,
            cp=candidate.cp,
            ep=candidate.ep,
            vpp=candidate.vpp,
            global_batch_size=16,
            micro_batch_size=1,
            sequence_length=4096,
        )

        # 验证配置
        is_valid, errors = validate_candidate(config, model_spec, hardware)

        # 生成 Region 布局
        regions = None
        if is_valid:
            try:
                regions = create_regions_for_candidate(
                    model_spec, config, strategy="bubble_optimized"
                )
            except Exception as e:
                is_valid = False
                errors.append(f"Region 生成失败: {str(e)}")

        result = {
            "name": candidate.name,
            "config": config.to_dict(),
            "is_valid": is_valid,
            "validation_errors": errors,
            "num_regions": len(regions) if regions else 0,
        }

        results.append(result)

        # 打印结果
        status = "✓" if is_valid else "✗"
        print(f"\n{status} {candidate.name}")
        print(f"  配置: TP={config.tp}, PP={config.pp}, DP={config.dp}, VPP={config.vpp}")
        print(f"  Microbatches: {config.num_microbatches}")

        if is_valid:
            print(f"  ✓ 验证通过, {len(regions)} 个 Region")
        else:
            print(f"  ✗ 验证失败:")
            for error in errors:
                print(f"    - {error}")

    # 保存结果
    output_dir = Path("./experiments/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n验证结果已保存到 {output_dir / 'validation_results.json'}")

    # 统计
    valid_count = sum(1 for r in results if r["is_valid"])
    print(f"\n【统计】")
    print(f"  总候选数: {len(results)}")
    print(f"  合法候选: {valid_count}")
    print(f"  不合法候选: {len(results) - valid_count}")

    return results


def run_region_comparison_test():
    """Region 分配策略对比测试"""
    print("\n" + "=" * 80)
    print("ParaMorph Region 分配策略对比测试")
    print("=" * 80)

    model_spec = ModelSpec(
        name="Lightning-30B",
        num_layers=52,
        hidden_size=6144,
        num_attention_heads=48,
        layer_types=[LayerType.ATTENTION] * 52,
        layer_order=list(range(52)),
    )

    # 候选 A: TP=2, PP=8, VPP=2
    config_a = ParallelConfig(
        tp=2, pp=8, dp=1, vpp=2,
        global_batch_size=16,
        micro_batch_size=1,
        sequence_length=4096,
    )

    print(f"\n【候选 A: TP=2, PP=8, VPP=2】")

    # 比较不同策略
    strategies = ["uniform", "bubble_optimized"]

    for strategy in strategies:
        regions = create_regions_for_candidate(model_spec, config_a, strategy=strategy)

        print(f"\n策略: {strategy}")
        print(f"Region 分配:")

        for pp_rank in range(config_a.pp):
            pp_regions = [r for r in regions if r.pp_rank == pp_rank]
            layer_dist = [r.num_layers for r in pp_regions]
            print(f"  PP Rank {pp_rank}: {layer_dist}")

        # 统计
        layer_counts = [r.num_layers for r in regions]
        print(f"  最少层数: {min(layer_counts)}, 最多层数: {max(layer_counts)}")

    return True


if __name__ == "__main__":
    # 运行完整验证测试
    validation_results = run_complete_validation_test()

    # 运行 Region 对比测试
    run_region_comparison_test()

    print("\n" + "=" * 80)
    print("所有测试完成")
    print("=" * 80)
