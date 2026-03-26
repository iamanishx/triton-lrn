# Triton Mental Model

## The short version

- `Triton` is for writing custom GPU kernels in Python syntax.
- `PyTorch` is for creating and managing tensors that live on the GPU.
- `CUDA` is the platform/driver layer that lets NVIDIA GPUs run that work.

Think of it like this:

- `CUDA`: the road and traffic system
- `PyTorch`: the truck carrying data around
- `Triton`: the custom engine you build for one job

## What Triton actually does

Triton lets you write code like:

```python
@triton.jit
def add_kernel(...):
    ...
```

That kernel is not normal Python code. Triton compiles it into GPU code and launches it on the device.

Triton is great for:

- vector add
- softmax
- layer norm
- attention pieces
- matrix multiplication kernels
- other performance-critical tensor operations

It is mainly a kernel language + compiler + launcher.

## Why PyTorch is needed

Triton does not try to be a full tensor library.

In your example, PyTorch is doing the host-side work:

- allocate GPU memory
- create input tensors
- create output tensors
- track shape, dtype, and device
- provide pointers/buffers for Triton to read and write
- make it easy to print and compare results

So this split is the key idea:

- `PyTorch` owns the tensors
- `Triton` computes on those tensors

Without PyTorch, you would need some other way to create GPU buffers and pass their addresses into Triton. That is possible, but it is not the easiest way to learn.

## What happens when `main.py` runs

1. Python starts your script.
2. PyTorch creates tensors on the GPU, like `x` and `y`.
3. You launch the Triton kernel with a grid.
4. Triton compiles the kernel the first time it sees that configuration.
5. The GPU runs many kernel instances in parallel.
6. Each kernel instance processes one block of elements.
7. The result is written into the output PyTorch tensor.

## How to read a Triton kernel mentally

For a simple vector add kernel:

- `pid = tl.program_id(axis=0)` means "which block am I?"
- `offsets = block_start + tl.arange(0, BLOCK_SIZE)` means "which elements does this block own?"
- `mask = offsets < n_elements` means "avoid reading past the end"
- `tl.load(...)` reads from GPU memory
- `tl.store(...)` writes to GPU memory

So a Triton kernel usually means:

- find my block
- compute my indices
- load data
- do math
- store data

## What more is needed besides Triton

For a beginner Triton setup, you usually need all of these:

### 1. Python environment

- `uv` is fine for this
- your project already has that part working

### 2. Triton package

- you already added `triton-windows<3.7`

### 3. PyTorch

- needed for GPU tensors in your current example
- install with `uv add torch`

### 4. NVIDIA GPU

- Triton targets NVIDIA-style GPU execution paths in setups like yours
- no CUDA-capable NVIDIA GPU means this example will not run as intended

### 5. NVIDIA driver

- this is the important runtime requirement
- if the driver is missing or broken, GPU code will not launch

### 6. CUDA-compatible PyTorch build

- `torch.cuda.is_available()` must return `True`
- if it returns `False`, PyTorch is installed but cannot use your GPU

### 7. Matching versions

- driver, PyTorch, CUDA runtime expectations, and Triton need to play nicely together
- most problems come from version mismatch rather than bad code

## Important nuance about CUDA

When people say "I installed CUDA," that can mean different things:

- CUDA driver/runtime support is available
- CUDA toolkit is installed
- PyTorch was installed with CUDA support

Those are related, but not identical.

For your Python script, the most important checks are:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If that prints `True`, you are much closer.

## Minimal beginner stack

For this repo, the mental model is:

- `Python` writes the host code
- `PyTorch` creates CUDA tensors
- `Triton` runs custom kernels on those tensors
- `NVIDIA driver/GPU` executes the work

## What you do not need right away

You do not need deep CUDA C++ knowledge to start learning Triton.

You also usually do not need to write raw pointer management code yourself when using Triton with PyTorch.

## First learning path

The natural order is:

1. get `torch.cuda.is_available()` to return `True`
2. run vector add
3. compare Triton output with PyTorch output
4. learn grid, `program_id`, `tl.arange`, `tl.load`, `tl.store`, and masks
5. move to softmax or matmul

## One-sentence mental model

PyTorch gives Triton some GPU tensors, Triton runs a custom kernel over them, and CUDA plus the GPU make the whole thing execute fast.
