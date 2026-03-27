# triton-lrn

Small learning repo for writing first Triton kernels on Windows.

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
