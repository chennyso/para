from __future__ import annotations

from dataclasses import dataclass, replace

from megatron.core.frontier.ir.boundary import BoundaryOp, BoundaryOpKind, BoundaryProgram
from megatron.core.frontier.ir.contract import PlacementKind, ValueContract


@dataclass(frozen=True)
class ConsumerDemand:
    shard_dim: int
    shard_start: int
    shard_end: int

    def __post_init__(self) -> None:
        if self.shard_dim < 0 or self.shard_start < 0 or self.shard_end <= self.shard_start:
            raise ValueError("consumer demand must contain a non-empty interval")


def can_factorize_reduce(resolve_op: BoundaryOp, demand: ConsumerDemand) -> bool:
    placement = resolve_op.input_contract.placement
    shape = resolve_op.input_contract.logical_shape
    return (placement.kind is PlacementKind.PARTIAL and placement.reduce_op == "sum"
            and demand.shard_dim < len(shape) and isinstance(shape[demand.shard_dim], int)
            and demand.shard_end <= shape[demand.shard_dim])


def factorize_reduce(program: BoundaryProgram, resolve_index: int, demand: ConsumerDemand) -> BoundaryProgram:
    resolve = program.ops[resolve_index]
    if resolve.kind is not BoundaryOpKind.RESOLVE:
        raise ValueError("factorization requires a resolve operation")
    if not can_factorize_reduce(resolve, demand):
        raise ValueError("resolve operation cannot satisfy this consumer demand")
    destination = program.destination_contract
    if destination.placement.kind is not PlacementKind.SHARD or destination.placement.dim != demand.shard_dim:
        raise ValueError("projection demand must match the consumer shard dimension")
    projected_slice = (demand.shard_dim, demand.shard_start, demand.shard_end)
    projection_contract = replace(resolve.input_contract, logical_slice=projected_slice,
                                  adjoint_key="project_scatter_add")
    projection = BoundaryOp(
        op_id=f"{resolve.op_id}:project",
        kind=BoundaryOpKind.PROJECT,
        input_contract=resolve.input_contract,
        output_contract=projection_contract,
        implementation_key="demand_projection",
    )
    local_output = replace(resolve.output_contract, logical_slice=projected_slice)
    local_resolve = BoundaryOp(
        op_id=f"{resolve.op_id}:local",
        kind=BoundaryOpKind.RESOLVE,
        input_contract=projection_contract,
        output_contract=local_output,
        implementation_key="local_resolve",
    )
    tail = []
    current = local_output
    for operation in program.ops[resolve_index + 1:]:
        output = replace(operation.output_contract, logical_slice=projected_slice)
        tail.append(replace(operation, input_contract=current, output_contract=output))
        current = output
    ops = program.ops[:resolve_index] + (projection, local_resolve) + tuple(tail)
    return replace(program, ops=ops, destination_contract=current, executable_witness_key=None)
