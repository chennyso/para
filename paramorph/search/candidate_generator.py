"""
Candidate Generator - Generate optimization candidates from actions
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import hashlib
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_data_models import Candidate, ParallelConfig, ModelSpec, HardwareTopology


class CandidateGenerator:
    """
    Generate candidate configurations from optimization actions

    Implements parameter closure: when changing one parameter,
    automatically adjust dependent parameters to maintain constraints
    """

    def __init__(self, model: ModelSpec, topology: HardwareTopology):
        self.model = model
        self.topology = topology

    def generate_from_baseline(
        self,
        baseline: ParallelConfig
    ) -> List[Candidate]:
        """
        Generate standard candidate set from baseline

        Returns predefined candidates A-E as specified in task requirements
        """
        candidates = []

        # Baseline
        baseline_cand = Candidate(
            candidate_id="baseline",
            parent_id=None,
            parallel_config=baseline,
            region_layout=None,
            optimization_actions=["baseline"],
            is_valid=True
        )
        candidates.append(baseline_cand)

        # Candidate A: VPP=2 (non-uniform region)
        config_a = ParallelConfig(
            tp=baseline.tp,
            pp=baseline.pp,
            dp=baseline.dp,
            vpp=2,
            global_batch_size=baseline.global_batch_size,
            micro_batch_size=baseline.micro_batch_size,
            sequence_length=baseline.sequence_length,
        )
        cand_a = Candidate(
            candidate_id="candidate_a",
            parent_id="baseline",
            parallel_config=config_a,
            region_layout=None,
            optimization_actions=["increase_vpp", "non_uniform_region"],
            is_valid=True
        )
        candidates.append(cand_a)

        # Candidate B: VPP=4 (fine-grained region)
        config_b = ParallelConfig(
            tp=baseline.tp,
            pp=baseline.pp,
            dp=baseline.dp,
            vpp=4,
            global_batch_size=baseline.global_batch_size,
            micro_batch_size=baseline.micro_batch_size,
            sequence_length=baseline.sequence_length,
        )
        cand_b = Candidate(
            candidate_id="candidate_b",
            parent_id="baseline",
            parallel_config=config_b,
            region_layout=None,
            optimization_actions=["increase_vpp_4x", "fine_grained_region"],
            is_valid=True
        )
        candidates.append(cand_b)

        # Candidate C: TP=4, PP=4, VPP=2 (joint optimization)
        config_c = ParallelConfig(
            tp=4,
            pp=4,
            dp=1,
            vpp=2,
            global_batch_size=baseline.global_batch_size,
            micro_batch_size=baseline.micro_batch_size,
            sequence_length=baseline.sequence_length,
        )
        cand_c = Candidate(
            candidate_id="candidate_c",
            parent_id="baseline",
            parallel_config=config_c,
            region_layout=None,
            optimization_actions=["increase_tp", "reduce_pp", "increase_vpp"],
            is_valid=True
        )
        candidates.append(cand_c)

        # Candidate D: TP=2, PP=4, DP=2, VPP=2, M=8
        config_d = ParallelConfig(
            tp=2,
            pp=4,
            dp=2,
            vpp=2,
            global_batch_size=baseline.global_batch_size,
            micro_batch_size=2,  # Increased from 1
            sequence_length=baseline.sequence_length,
        )
        cand_d = Candidate(
            candidate_id="candidate_d",
            parent_id="baseline",
            parallel_config=config_d,
            region_layout=None,
            optimization_actions=["reduce_pp", "increase_dp", "increase_vpp", "increase_mbs"],
            is_valid=True
        )
        candidates.append(cand_d)

        # Candidate E: TP=4, PP=2, DP=2, VPP=1, M=8
        config_e = ParallelConfig(
            tp=4,
            pp=2,
            dp=2,
            vpp=1,
            global_batch_size=baseline.global_batch_size,
            micro_batch_size=2,
            sequence_length=baseline.sequence_length,
        )
        cand_e = Candidate(
            candidate_id="candidate_e",
            parent_id="baseline",
            parallel_config=config_e,
            region_layout=None,
            optimization_actions=["increase_tp", "reduce_pp_minimal", "increase_dp", "increase_mbs"],
            is_valid=True
        )
        candidates.append(cand_e)

        return candidates

    def apply_parameter_closure(
        self,
        base_config: ParallelConfig,
        changes: Dict[str, int]
    ) -> ParallelConfig:
        """
        Apply parameter changes with automatic closure

        When changing TP/PP/DP, automatically recompute to maintain world_size
        """
        # Start with base values
        tp = changes.get("tp", base_config.tp)
        pp = changes.get("pp", base_config.pp)
        dp = changes.get("dp", base_config.dp)
        cp = changes.get("cp", base_config.cp)
        ep = changes.get("ep", base_config.ep)
        vpp = changes.get("vpp", base_config.vpp)

        # Maintain world size constraint
        target_world_size = self.topology.num_gpus

        # If TP/PP/CP/EP specified but not DP, compute DP
        if "dp" not in changes:
            divisor = tp * pp * cp
            if target_world_size % divisor:
                raise ValueError('TP乘PP乘CP不能整除实际GPU数量')
            dp = target_world_size // divisor

        # Verify world size
        computed_world_size = tp * pp * dp * cp
        if computed_world_size != target_world_size:
            raise ValueError(
                f"Parameter closure failed: "
                f"TP={tp} * PP={pp} * DP={dp} * CP={cp} = {computed_world_size} "
                f"!= {target_world_size}"
            )

        # Recompute microbatch count if MBS changed
        mbs = changes.get("micro_batch_size", base_config.micro_batch_size)
        # M = GBS / (DP * MBS)

        return base_config.with_updates(tp=tp, pp=pp, dp=dp, cp=cp, ep=ep, vpp=vpp,
            etp=changes.get('etp', tp), micro_batch_size=mbs, use_sequence_parallel=tp > 1)

    def generate_candidate_id(self, config: ParallelConfig) -> str:
        """Generate unique ID from config"""
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        hash_val = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"cand_{hash_val}"
