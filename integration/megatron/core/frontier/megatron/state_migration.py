import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.frontier.planner.verifier import json_value


def local_geometry(contract, rank):
    if contract.storage_intervals:
        raise ValueError('Flattened native optimizer storage requires its interval-aware migration implementation')
    if rank not in contract.storage_owners:
        raise ValueError('The requested rank does not own this training state')
    coordinates = {}
    remainder = contract.physical_ranks.index(rank)
    for axis, size in reversed(contract.logical_mesh):
        coordinates[axis] = remainder % size
        remainder //= size
    sizes = dict(contract.logical_mesh)
    fragments, offsets = {}, {}
    for axis, dimension in contract.sharding:
        offsets[dimension] = offsets.get(dimension, 0) * sizes[axis] + coordinates[axis]
        fragments[dimension] = fragments.get(dimension, 1) * sizes[axis]
    shape = tuple(size // fragments.get(dimension, 1) for dimension, size in enumerate(contract.value.logical_shape))
    rank_offsets = tuple((dimension, offsets[dimension], fragments[dimension]) for dimension in sorted(fragments))
    peers = [owner for owner in contract.storage_owners if shard_coordinate(contract, owner) == shard_coordinate(contract, rank)]
    return shape, rank_offsets, peers.index(rank)


def shard_coordinate(contract, rank):
    remainder = contract.physical_ranks.index(rank)
    coordinates = {}
    for axis, size in reversed(contract.logical_mesh):
        coordinates[axis] = remainder % size
        remainder //= size
    return tuple(coordinates[axis] for axis, _ in contract.sharding)


def complete_storage(contract):
    sizes = dict(contract.logical_mesh)
    expected = set(product(*(range(sizes[axis]) for axis, _ in contract.sharding)))
    return {shard_coordinate(contract, rank) for rank in contract.storage_owners} == expected


def training_tensors(parameter):
    states = [('parameter', parameter.parameter), ('gradient', parameter.gradient)]
    states.extend(('optimizer:' + value.value.logical_value, value) for value in parameter.optimizer_states)
    if len({name for name, _ in states}) != len(states):
        raise ValueError('Optimizer state identities must be distinct within each parameter')
    return {parameter.parameter_key + '/' + name: value for name, value in states}


@dataclass(frozen=True)
class MigrationReceipt:
    backend: str
    plan_id: str
    rank: int
    global_ranks: tuple[int, ...]
    loaded_keys: tuple[str, ...]
    checkpoint_directory: str
    save_seconds: float
    load_seconds: float
    publish_seconds: float


class MegatronCheckpointStateMigration:
    key = 'megatron.torch_dist.training_state'

    def contracts(self, migrations):
        source, destination = {}, {}
        for migration in migrations:
            old = training_tensors(migration.source)
            new = training_tensors(migration.destination)
            if set(source) & set(old) or set(destination) & set(new):
                raise ValueError('Each training state must have exactly one migration owner')
            source.update(old)
            destination.update(new)
        return source, destination

    def supports(self, migrations):
        source, destination = self.contracts(migrations)
        if not migrations or source.keys() != destination.keys():
            return False
        for key, old in source.items():
            new = destination[key]
            if old.value.logical_shape != new.value.logical_shape or old.value.dtype != new.value.dtype:
                return False
            for value in (old, new):
                if (value.storage_intervals or value.partial or value.layout != 'contiguous' or value.value.logical_slice is not None
                        or value.value.presence_policy != 'required'
                        or any(not isinstance(size, int) for size in value.value.logical_shape)
                        or not complete_storage(value)):
                    return False
        return True

    def plan_id(self, migrations):
        descriptor = {'backend': self.key, 'migrations': [asdict(migration) for migration in migrations]}
        return hashlib.sha256(json.dumps(descriptor, sort_keys=True, default=json_value, allow_nan=False).encode()).hexdigest()

    def validate_bindings(self, contracts, bindings, rank):
        expected = {key for key, value in contracts.items() if rank in value.storage_owners}
        if set(bindings) != expected:
            raise ValueError('Actual training tensors differ from the complete local migration inventory')
        for key in expected:
            tensor, contract = bindings[key], contracts[key]
            shape, _, _ = local_geometry(contract, rank)
            if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                    or str(tensor.dtype).removeprefix('torch.') != contract.value.dtype
                    or not tensor.is_contiguous()):
                raise ValueError('Actual tensor metadata differs from its migration contract: ' + key)

    def sharded_state(self, contracts, bindings, rank):
        result = {}
        for key in sorted(bindings):
            shape, offsets, replica = local_geometry(contracts[key], rank)
            tensor = bindings[key].detach()
            if not shape:
                tensor = tensor.reshape(1)
            result[key] = ShardedTensor.from_rank_offsets(key, tensor, *offsets, replica_id=replica)
        return result

    def migrate(self, migrations, source_bindings, destination_bindings, checkpoint_directory,
            global_group, process_groups, update_boundary):
        if not update_boundary:
            raise ValueError('Training state migration requires a completed update boundary')
        if not self.supports(migrations):
            raise ValueError('Unsupported training state migration geometry')
        if not dist.is_initialized() or global_group is None:
            raise ValueError('Training state migration requires an actual initialized global process group')
        ranks = tuple(dist.get_process_group_ranks(global_group))
        if ranks != tuple(range(dist.get_world_size())) or dist.get_backend(global_group) != 'nccl':
            raise ValueError('Megatron checkpoint migration requires the actual complete NCCL world group')
        rank = dist.get_rank()
        source, destination = self.contracts(migrations)
        for migration in migrations:
            for parameter in (migration.source, migration.destination):
                for group in (*parameter.gradient_reduce_groups, *parameter.replica_groups):
                    if rank in group.ranks:
                        handle = process_groups[group.key]
                        if (tuple(dist.get_process_group_ranks(handle)) != group.ranks
                                or dist.get_backend(handle) != group.backend):
                            raise ValueError('Actual state synchronization group differs from the migration contract')
        if any(not set(value.physical_ranks) <= set(ranks) for value in (*source.values(), *destination.values())):
            raise ValueError('The global migration group does not cover every source and destination rank')
        self.validate_bindings(source, source_bindings, rank)
        self.validate_bindings(destination, destination_bindings, rank)
        plan = self.plan_id(migrations)
        descriptors = [None] * len(ranks)
        dist.all_gather_object(descriptors, (plan, str(Path(checkpoint_directory).resolve())), group=global_group)
        if len(set(descriptors)) != 1:
            raise ValueError('All ranks must execute one identical state migration plan and directory')
        directory = Path(checkpoint_directory)
        if rank == 0:
            directory.mkdir(parents=True, exist_ok=False)
        dist.barrier(group=global_group)
        torch.cuda.synchronize()
        started = perf_counter()
        dist_checkpointing.save(self.sharded_state(source, source_bindings, rank), str(directory),
            sharded_strategy=('torch_dist', 1), validate_access_integrity=True)
        save_seconds = perf_counter() - started
        staging = {key: torch.empty_like(value) for key, value in destination_bindings.items()}
        started = perf_counter()
        loaded = dist_checkpointing.load(self.sharded_state(destination, staging, rank), str(directory),
            validate_access_integrity=True, strict='raise_all')
        if set(loaded) != set(staging):
            raise ValueError('The checkpoint loader returned an incomplete training state inventory')
        loaded = {key: value.reshape(destination_bindings[key].shape) for key, value in loaded.items()}
        self.validate_bindings(destination, loaded, rank)
        torch.cuda.synchronize()
        load_seconds = perf_counter() - started
        dist.barrier(group=global_group)
        started = perf_counter()
        with torch.no_grad():
            for key in sorted(destination_bindings):
                destination_bindings[key].copy_(loaded[key])
        torch.cuda.synchronize()
        dist.barrier(group=global_group)
        publish_seconds = perf_counter() - started
        return MigrationReceipt(self.key, plan, rank, ranks, tuple(sorted(loaded)), str(directory),
            save_seconds, load_seconds, publish_seconds)
