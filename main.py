import importlib
import importlib.util

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x, y):
    torch = importlib.import_module("torch")

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)  # type: ignore[arg-type]
    return out


def main():
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        raise SystemExit("This example needs PyTorch. Install it with `uv add torch`.")

    torch = importlib.import_module("torch")

    if not torch.cuda.is_available():
        raise SystemExit(
            "This Triton example needs a CUDA-capable GPU and a CUDA-enabled PyTorch install."
        )

    x = torch.arange(8, device="cuda", dtype=torch.float32)
    y = torch.ones_like(x)
    out = add(x, y)

    print("x:", x)
    print("y:", y)
    print("out:", out)


if __name__ == "__main__":
    main()
