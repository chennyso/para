from __future__ import annotations

from dataclasses import replace

from megatron.core.frontier.ir.boundary import BoundaryOp, BoundaryOpKind, BoundaryProgram
from megatron.core.frontier.ir.contract import PlacementKind, ValueContract


def _resolved_contract(contract: ValueContract) -> ValueContract:
    obligation = contract.obligation
    if obligation is None:
        return contract
    return replace(contract, placement=obligation.target, obligation=None)


def _with_placement(contract: ValueContract, placement) -> ValueContract:
    obligation = None
    if placement.kind is PlacementKind.PARTIAL:
        raise ValueError("canonical repartition cannot target a partial placement")
    return replace(contract, placement=placement, obligation=obligation)


def build_canonical_boundary(
    boundary_id: int,
    source_region: int,
    destination_region: int,
    source_contract: ValueContract,
    destination_contract: ValueContract,
) -> BoundaryProgram:
    ops: list[BoundaryOp] = []
    current = source_contract
    if current.placement.kind is PlacementKind.PARTIAL or current.obligation is not None:
        resolved = _resolved_contract(current)
        ops.append(BoundaryOp(
            op_id=f"{boundary_id}:resolve",
            kind=BoundaryOpKind.RESOLVE,
            input_contract=current,
            output_contract=resolved,
            implementation_key="nccl_resolve",
        ))
        current = resolved
    if current.placement != destination_contract.placement or current.mesh_key != destination_contract.mesh_key:
        repartitioned = _with_placement(current, destination_contract.placement)
        repartitioned = replace(repartitioned, mesh_key=destination_contract.mesh_key)
        ops.append(BoundaryOp(
            op_id=f"{boundary_id}:repartition",
            kind=BoundaryOpKind.REPARTITION,
            input_contract=current,
            output_contract=repartitioned,
            implementation_key="standard_repartition",
        ))
        current = repartitioned
    ops.append(BoundaryOp(
        op_id=f"{boundary_id}:transfer",
        kind=BoundaryOpKind.TRANSFER,
        input_contract=current,
        output_contract=destination_contract,
        implementation_key="pipeline_transfer",
    ))
    return BoundaryProgram(
        boundary_id=boundary_id,
        source_region=source_region,
        destination_region=destination_region,
        source_contract=source_contract,
        destination_contract=destination_contract,
        ops=tuple(ops),
        transferred_bytes=_item_size(destination_contract.dtype) * _numel(destination_contract.logical_shape),
    )


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        if not isinstance(value, int) or value < 1:
            raise ValueError("canonical transfer requires a concrete compiled payload shape")
        result *= value
    return result


def _item_size(dtype):
    return {"float32": 4, "bfloat16": 2, "float16": 2, "int32": 4, "int64": 8, "bool": 1}[dtype]
