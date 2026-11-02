import argparse
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import unittest

from core_data_models import HardwareTopology
from graph.capability_registry import CapabilityRegistry
from graph.complete_rewrite import CompleteAndRewrite
from graph.model_graph import read_model
from predictor.profile_database import normalized_execution
from search.conditional_partition import ConditionalPartitionSolver, GLOBAL_FIELDS, strictly_dominates
from search.seed_planner import SeedPlanner
from search.family_state import FamilyLedger
from storage.experiment_store import ExperimentStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    request = config['request']
    model, graph = read_model(config['model_config_path'])
    paths = config.get('hardware_topology_paths') or [args.config.parent / 'hardware-topology.json']
    hardware = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        hardware.extend(data if isinstance(data, list) else [data])
    devices = [dict(device, node_index=index, physical_rank=index * request['gpus_per_node'] + device['index'])
        for index, node in enumerate(hardware) for device in node['devices']]
    world = request['node_count'] * request['gpus_per_node']
    assert len(devices) == world
    topology = HardwareTopology(world, request['node_count'], request['gpus_per_node'],
        min(row['capacity_bytes'] for row in devices) / 2**30, devices=devices,
        software=hardware[0]['software'])
    connection = sqlite3.connect(f'file:{Path(config["profile_database"]).resolve()}?mode=ro', uri=True)
    profiles = [json.loads(row[0]) for row in connection.execute('SELECT document FROM profiles')]
    normalized = normalized_execution(request)
    reference = next(row for row in profiles if row['request']['model'] == request['model']
        and all(normalized_execution(row['request'])[key] == normalized[key]
            for key in ('tp', 'cp', 'ep', 'etp', 'micro_batch', 'sequence', 'recompute')))
    registry = CapabilityRegistry(graph['module_structure'])
    domains = registry.parallel_domains(model, topology, request)
    rewriter = CompleteAndRewrite(model, topology, registry, domains)
    solver = ConditionalPartitionSolver(rewriter, profiles,
        backend_source_hashes=reference['source_hashes'])
    parent_values = SeedPlanner(rewriter).global_family_parent(request)
    global_values = {key: parent_values[key] for key in GLOBAL_FIELDS}
    assert 2 in domains['pp']
    global_values.update(pp=2, dp=world // (request['tp'] * request['cp'] * 2),
        expert_dp=world // (global_values['etp'] * global_values['ep'] * 2))
    fixed = dict(vpp=1, micro_batch=request['micro_batch'], recompute=request['recompute'],
        execution_backend='megatron_ddp', optimizer_sharding='distributed',
        group=1, overlap_p2p=False, pipeline_schedule='1f1b')
    exhaustive = solver.solve(request, global_values, reference['id'], fixed_values=fixed,
        limit=model.num_layers, max_layouts=1, max_states_per_boundary=model.num_layers,
        max_expansions=model.num_layers ** 3)
    expected = {(cut, model.num_layers - cut) for cut in range(1, model.num_layers)}
    actual = {tuple(row['request']['decoder_chunks']) for row in exhaustive['candidates']}
    assert actual == expected
    assert exhaustive['search_complete']
    assert exhaustive['status'] == 'UNKNOWN'
    assert exhaustive['lower_bound_ms'] is None
    assert exhaustive['layer_load_reference_evidence']['admissible_replay_lower_bound'] is False
    assert all(row['partition_evidence']['compute_profile_id'] for row in exhaustive['candidates'])
    assert not exhaustive['missing_cost_requests']
    profile = solver._profile(exhaustive['candidates'][0]['request'])
    layer_times = [profile['layers'][str(index)]['forward_ms'] + profile['layers'][str(index)]['backward_ms']
        for index in range(model.num_layers)]
    def objective(counts):
        cut = counts[0]
        return max(sum(layer_times[:cut]), sum(layer_times[cut:]))
    best = min(objective(counts) for counts in expected)
    assert abs(objective(exhaustive['candidates'][0]['request']['decoder_chunks']) - best) < 1e-8
    first = exhaustive['candidates'][0]
    compiled = rewriter.complete_and_rewrite(request, first['intent'], limit=1)[0]
    assert compiled['execution_graph']['tasks']
    assert all(dependency in compiled['execution_graph']['tasks']
        for task in compiled['execution_graph']['tasks'].values() for dependency in task['dependencies'])
    states, _ = solver._partition(first['request'], profile, model.num_layers, model.num_layers ** 3)
    assert not strictly_dominates(states[0], states[0])
    different_boundary = replace(states[0], boundary=states[1].boundary + ('different',))
    assert not strictly_dominates(states[0], different_boundary)
    assert not strictly_dominates(states[0], states[1])
    truncated = solver.solve(request, global_values, reference['id'], fixed_values=fixed,
        limit=3, max_layouts=1, max_states_per_boundary=1, max_expansions=8)
    assert not truncated['search_complete']
    assert truncated['status'] == 'UNKNOWN'
    assert truncated['candidates']
    assert 'PARTITION_BUDGET' in truncated['termination']
    selected_partitions = [list(counts) for counts in sorted(actual)[:2]]
    restricted = solver.solve(request, global_values, reference['id'], fixed_values=fixed,
        allowed_ranges=dict(decoder_chunks=selected_partitions),
        preserve=['tp', 'cp', 'micro_batch', 'recompute'], limit=2, max_layouts=2,
        max_states_per_boundary=2, max_expansions=64)
    assert {tuple(row['request']['decoder_chunks']) for row in restricted['candidates']} == {
        tuple(counts) for counts in selected_partitions}
    assert restricted['lower_bound_ms'] is None
    check = unittest.TestCase()
    with check.assertRaises(ValueError):
        solver.solve(request, global_values, reference['id'], fixed_values=dict(fixed,
            micro_batch=request['micro_batch'] + 1), preserve=['micro_batch'])
    with check.assertRaises(ValueError):
        solver.solve(request, global_values, reference['id'], fixed_values=dict(fixed,
            precision_aware_optimizer=not request.get('precision_aware_optimizer', False)))
    with check.assertRaises(ValueError):
        solver.solve(request, global_values, reference['id'], fixed_values=fixed,
            allowed_ranges=dict(vpp=[2]))
    unsatisfied = solver.solve(request, global_values, reference['id'],
        fixed_values=dict(fixed, pipeline_schedule='interleaved_1f1b'))
    assert unsatisfied['status'] == 'UNKNOWN'
    assert unsatisfied['conditional_status'] == 'INFEASIBLE'
    assert not unsatisfied['candidates']
    interleaved = solver.solve(request, global_values, reference['id'],
        fixed_values=dict(fixed, vpp=2, group=2, pipeline_schedule='interleaved_1f1b'),
        limit=4, max_layouts=1, max_states_per_boundary=4, max_expansions=4000)
    assert interleaved['candidates']
    for row in interleaved['candidates']:
        assert len(row['request']['decoder_chunks']) == 4
        assert row['request']['vpp'] == 2 and row['request']['group'] == 2
        assert row['request']['execution_backend'] == fixed['execution_backend']
        assert [region['pp_rank'] for region in row['partition_evidence']['region_implementations']] == [0, 1, 0, 1]
        assert [region['vp_rank'] for region in row['partition_evidence']['region_implementations']] == [0, 0, 1, 1]
    interleaved_graph = rewriter.complete_and_rewrite(request,
        interleaved['candidates'][0]['intent'], limit=1)[0]['execution_graph']
    assert all(dependency in interleaved_graph['tasks']
        for task in interleaved_graph['tasks'].values() for dependency in task['dependencies'])
    no_profile = ConditionalPartitionSolver(rewriter).solve(request, global_values, reference['id'],
        fixed_values=fixed, limit=2, max_layouts=1, max_expansions=8)
    assert no_profile['status'] == 'UNKNOWN'
    assert no_profile['missing_cost_requests']
    assert all(row['partition_evidence']['stage_forward_ms'] is None for row in no_profile['candidates'])
    impossible = dict(global_values, dp=world)
    proof = solver.solve(request, impossible, reference['id'], fixed_values=fixed)
    assert proof['status'] == 'INFEASIBLE' and proof['proofs'] and not proof['candidates']
    initial_planner = SeedPlanner(rewriter)
    initial = initial_planner.random_initial_plan(request, 0)
    assert initial['execution_backend'] in request['allowed_execution_backends']
    assert initial['optimizer_sharding'] in request['allowed_optimizer_sharding']
    assert initial_planner.random_initial_selection['prediction_used'] is False
    assert initial_planner.random_initial_selection['measurement_used'] is False
    store = ExperimentStore(args.output.with_suffix('.families.sqlite'))
    families = FamilyLedger(store)
    family_request = dict(request, **global_values)
    recorded = families.record_solver(family_request,
        dict(restricted, scope='conditional_search_domain'), reference['id'])
    assert recorded['lower_bound_ms'] is None
    recorded = families.record_solver(family_request,
        dict(unsatisfied, scope='conditional_search_domain'), reference['id'])
    assert recorded['status'] == 'UNKNOWN'
    assert recorded['lower_bound_ms'] is None
    report = dict(status='PASS', scope='existing_real_profile_cpu_partition_validation',
        model=model.name, gpu_count=world, profile_id=profile['id'],
        exhaustive_candidate_count=len(actual), exhaustive_matches_all_contiguous_cuts=True,
        best_layer_load_ms=best, best_counts=first['request']['decoder_chunks'],
        compiled_task_count=len(compiled['execution_graph']['tasks']),
        vpp_candidate_count=len(interleaved['candidates']), vpp_compiled_task_count=len(interleaved_graph['tasks']),
        truncated_candidate_count=len(truncated['candidates']), truncated_termination=truncated['termination'],
        unknown_without_profile=no_profile['status'], algebraic_proof=proof['proofs'],
        allowed_partitions_preserved=True, preserve_conflicts_rejected=True,
        optimizer_precision_fixed=True, conditional_infeasibility_keeps_family_unknown=True,
        unproven_global_lower_bound=None,
        initial_backend_coverage=sorted(backends), measured_gpu_benchmarks=0, llm_requests=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
