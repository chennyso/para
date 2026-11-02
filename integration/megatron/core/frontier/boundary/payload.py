from dataclasses import dataclass


@dataclass(frozen=True)
class PayloadField:
    name: str
    shape: tuple[int, ...]
    dtype: str
    placement: object
    gradient_eligible: bool
    optional: bool
    adjoint_key: str


@dataclass(frozen=True)
class BoundaryPayload:
    fields: tuple[PayloadField, ...]

    def __post_init__(self):
        names = tuple(field.name for field in self.fields)
        if not names or len(set(names)) != len(names):
            raise ValueError('boundary payload requires distinct fields')


def derive_payload(cut):
    return BoundaryPayload(tuple(PayloadField(contract.logical_value, contract.logical_shape, contract.dtype,
                                              contract.placement, contract.gradient_eligible,
                                              contract.presence_policy == 'optional', contract.adjoint_key)
                                 for contract in cut.source_contracts))
