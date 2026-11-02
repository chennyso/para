import torch
import triton
import triton.language as tl


@triton.jit
def reduce_bf16_four(output, first, second, third, fourth, count: tl.constexpr, block: tl.constexpr):
    indices = tl.program_id(0) * block + tl.arange(0, block)
    mask = indices < count
    accumulator = tl.load(first + indices, mask, other=0).to(tl.float32)
    accumulator = (accumulator + tl.load(second + indices, mask, other=0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    accumulator = (accumulator + tl.load(third + indices, mask, other=0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    accumulator = (accumulator + tl.load(fourth + indices, mask, other=0).to(tl.float32)).to(tl.bfloat16)
    tl.store(output + indices, accumulator, mask)


def reduce_projected_ring(output, contributions, order):
    if (len(contributions) != 4 or sorted(order) != [0, 1, 2, 3]
            or output.dtype != torch.bfloat16 or not output.is_cuda or not output.is_contiguous()
            or any(value.shape != output.shape or value.dtype != output.dtype or value.device != output.device
                   or not value.is_contiguous() for value in contributions)):
        raise ValueError('fused PROJECT reduction requires four actual contiguous BF16 contributions')
    inputs = tuple(contributions[index] for index in order)
    reduce_bf16_four[(triton.cdiv(output.numel(), 1024),)](output, *inputs, output.numel(), 1024, num_warps=4)
