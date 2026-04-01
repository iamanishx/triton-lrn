import importlib
import importlib.util

import triton
import triton.language as tl


@triton.jit
def add_one_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # `pid` tells us which Triton program instance we are.
    pid = tl.program_id(axis=0)

    # Each program instance handles one chunk of size `BLOCK_SIZE`.
    block_start = pid * BLOCK_SIZE

    # Build the exact indices this program instance owns.
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Protect reads/writes near the end of the tensor.
    mask = offsets < n_elements

    # Read a chunk from x.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Do a tiny bit of work: add 1.
    out = x + 1

    # Write the result back.
    tl.store(out_ptr + offsets, out, mask=mask)


def add_one(x):
    torch = importlib.import_module("torch")

    out = torch.empty_like(x)
    n_elements = x.numel()

    # Launch enough program instances to cover the full tensor.
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Here one program instance handles 256 elements.
    add_one_kernel[grid](x, out, n_elements, BLOCK_SIZE=256)  # type: ignore[arg-type]
    return out


def main():
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        raise SystemExit(
            "This example needs PyTorch. Install CUDA PyTorch first for this repo."
        )

    torch = importlib.import_module("torch")

    if not torch.cuda.is_available():
        raise SystemExit(
            "This Triton example needs a CUDA-capable GPU and a CUDA-enabled PyTorch install."
        )

    # A simple 1D tensor on the GPU.
    x = torch.arange(10, device="cuda", dtype=torch.float32)

    # Triton result.
    out_triton = add_one(x)

    # PyTorch result for correctness checking.
    out_torch = x + 1

    print("input      :", x)
    print("triton out :", out_triton)
    print("torch out  :", out_torch)
    print("match      :", torch.allclose(out_triton, out_torch))


if __name__ == "__main__":
    main()
