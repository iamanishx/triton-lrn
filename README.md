# triton-lrn

Small learning repo for writing first Triton kernels on Windows.

## What is in this repo

- `main.py`: first Triton vector-add kernel
- `mental_model.md`: notes on how Triton, PyTorch, and CUDA fit together

## Why `torch` is not in `pyproject.toml`

This project needs a CUDA-enabled PyTorch build, not the default CPU-only wheel.

If `torch` is listed as a normal dependency, a fresh environment may install CPU PyTorch again, which makes:

- `torch.cuda.is_available()` return `False`
- Triton examples fail at runtime

So `torch` is installed manually with the correct CUDA wheel.

## Setup

Install the project dependencies:

```powershell
uv sync
```

Install CUDA PyTorch for Windows:

```powershell
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch
```

## Verify the GPU stack

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected shape of the output:

- version includes `+cu128`
- CUDA version prints `12.8`
- `torch.cuda.is_available()` prints `True`

## Run the example

```powershell
uv run python main.py
```

## Next steps

1. Read `mental_model.md`
2. Run and tweak `main.py`
3. Add another kernel like softmax or matmul
