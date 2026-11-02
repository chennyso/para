import json
import os
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import fields
from core_data_models import Candidate, SearchHistory, Measurement, ParallelConfig, MemoryPolicy, Region
import hashlib


class HistoryManager:
    """
    搜索历史管理器

    职责：
    1. 记录所有候选配置
    2. 追踪真实训练结果
    3. 防止重复执行相同配置
    4. 支持历史反馈和成本模型更新
    """

    def __init__(self, history_dir: str = "./experiments/history"):
        self.history_dir = history_dir
        os.makedirs(history_dir, exist_ok=True)
        self.history_file = os.path.join(history_dir, "search_history.json")
        self.history = self._load_history()

    def _load_history(self) -> SearchHistory:
        """加载搜索历史"""
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                data = json.load(f)
                history = SearchHistory()
                history.best_candidate_id = data.get("best_candidate_id")
                history.best_throughput = data.get("best_throughput")
                history.total_gpu_hours = data.get("total_gpu_hours", 0.0)

                for c_data in data.get("candidates", []):
                    raw_config = c_data['config']
                    config_fields = {item.name for item in fields(ParallelConfig)}
                    config_values = {key: value for key, value in raw_config.items() if key in config_fields}
                    if 'memory_policy' in config_values:
                        config_values['memory_policy'] = MemoryPolicy(**config_values['memory_policy'])
                    config_values['device_mapping'] = tuple(config_values.get('device_mapping', ()))
                    config = ParallelConfig(**config_values)
                    offset, regions = 0, []
                    for row in c_data['region_layout']:
                        regions.append(Region(row['region_id'], row['pp_rank'], row['vp_rank'],
                            offset, offset + row['num_layers']))
                        offset += row['num_layers']
                    candidate = Candidate(c_data['candidate_id'], c_data['parent_id'], config,
                        regions or None, c_data['optimization_actions'])
                    for key in ('predicted_speedup', 'predicted_critical_path_ms', 'is_valid',
                        'validation_errors', 'actual_throughput', 'actual_iteration_time',
                        'execution_status', 'execution_plan_sha256', 'transformation_record'):
                        if key in c_data:
                            setattr(candidate, key, c_data[key])
                    history.candidates.append(candidate)

                return history
        else:
            return SearchHistory()

    def _save_history(self):
        """保存搜索历史"""
        with open(self.history_file, "w") as f:
            json.dump(self.history.to_dict(), f, indent=2)

    def generate_candidate_id(self, candidate: Candidate) -> str:
        """
        为候选配置生成唯一 ID

        使用配置内容的哈希值确保相同配置产生相同 ID
        """
        config_str = json.dumps(dict(config=candidate.parallel_config.to_dict(),
            regions=[(region.layer_start, region.layer_end, region.pp_rank, region.vp_rank)
                for region in candidate.region_layout or []],
            execution_plan_sha256=candidate.execution_plan_sha256), sort_keys=True)
        hash_obj = hashlib.sha256(config_str.encode())
        return f"cand_{hash_obj.hexdigest()[:12]}"

    def is_duplicate(self, candidate: Candidate) -> bool:
        """检查候选是否已经执行过"""
        candidate_id = self.generate_candidate_id(candidate)

        for existing in self.history.candidates:
            if self.generate_candidate_id(existing) == candidate_id:
                return True

        return False

    def add_candidate(
        self,
        candidate: Candidate,
        measurements: Optional[List[Measurement]] = None
    ) -> str:
        """
        添加候选配置到历史

        返回: candidate_id
        """
        # 生成候选 ID
        if not candidate.candidate_id:
            candidate.candidate_id = self.generate_candidate_id(candidate)

        # 添加到历史
        self.history.add_candidate(candidate)

        # 保存到磁盘
        self._save_history()

        # 保存详细的候选记录
        self._save_candidate_detail(candidate, measurements)

        return candidate.candidate_id

    def update_candidate_result(
        self,
        candidate_id: str,
        throughput: float,
        iteration_time: float,
        status: str = "completed",
        measurements: Optional[List[Measurement]] = None
    ):
        """
        更新候选的真实训练结果
        """
        for candidate in self.history.candidates:
            if candidate.candidate_id == candidate_id:
                candidate.actual_throughput = throughput
                candidate.actual_iteration_time = iteration_time
                candidate.execution_status = status

                # 更新最佳候选
                if throughput > (self.history.best_throughput or 0):
                    self.history.best_throughput = throughput
                    self.history.best_candidate_id = candidate_id

                break

        self._save_history()

        # 更新详细记录
        if measurements:
            self._save_candidate_detail(
                next(c for c in self.history.candidates if c.candidate_id == candidate_id),
                measurements
            )

    def _save_candidate_detail(
        self,
        candidate: Candidate,
        measurements: Optional[List[Measurement]] = None
    ):
        """保存候选的详细记录"""
        detail_file = os.path.join(
            self.history_dir, f"{candidate.candidate_id}_detail.json"
        )

        detail = {
            "candidate_id": candidate.candidate_id,
            "timestamp": datetime.now().isoformat(),
            "config": candidate.to_dict(),
            "measurements": [m.to_dict() for m in (measurements or [])],
        }

        with open(detail_file, "w") as f:
            json.dump(detail, f, indent=2)

    def get_best_candidate(self) -> Optional[Candidate]:
        """获取当前最佳候选"""
        if not self.history.best_candidate_id:
            return None

        for candidate in self.history.candidates:
            if candidate.candidate_id == self.history.best_candidate_id:
                return candidate

        return None

    def get_baseline(self) -> Optional[Candidate]:
        """获取 Baseline 候选"""
        for candidate in self.history.candidates:
            if "baseline" in candidate.candidate_id.lower():
                return candidate

        return None

    def generate_summary(self) -> Dict:
        """生成搜索历史摘要"""
        completed = [
            c for c in self.history.candidates
            if c.execution_status == "completed"
        ]

        valid = [c for c in completed if c.is_valid]
        invalid = [c for c in self.history.candidates if not c.is_valid]

        return {
            "total_candidates": len(self.history.candidates),
            "completed": len(completed),
            "valid": len(valid),
            "invalid": len(invalid),
            "best_candidate_id": self.history.best_candidate_id,
            "best_throughput": self.history.best_throughput,
            "total_gpu_hours": self.history.total_gpu_hours,
        }

    def export_csv(self, output_file: str):
        """导出历史记录为 CSV"""
        import csv

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)

            # 表头
            writer.writerow([
                "candidate_id",
                "tp", "pp", "dp", "cp", "ep", "vpp",
                "num_microbatches",
                "is_valid",
                "predicted_speedup",
                "actual_throughput",
                "actual_iteration_time",
                "execution_status",
            ])

            # 数据行
            for candidate in self.history.candidates:
                config = candidate.parallel_config
                writer.writerow([
                    candidate.candidate_id,
                    config.tp, config.pp, config.dp,
                    config.cp, config.ep, config.vpp,
                    config.num_microbatches,
                    candidate.is_valid,
                    candidate.predicted_speedup,
                    candidate.actual_throughput,
                    candidate.actual_iteration_time,
                    candidate.execution_status,
                ])
