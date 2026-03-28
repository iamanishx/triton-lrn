# Triton Basics to Intermediate

This note explains Triton from the point of view that matters when you are writing kernels.

It is not a full GPU architecture textbook. It is a practical mental model for someone who already knows some PyTorch and wants to understand how Triton maps work onto the GPU.

This write-up is based mainly on Triton's official programming guide and tutorials for vector add, fused softmax, and matrix multiplication, plus the PyTorch blog post on warp specialization.

## 1. What Triton is

Triton is a Python-embedded language and compiler for writing GPU kernels.

The important idea is:

- PyTorch gives you tensors
- Triton gives you custom kernels on those tensors
- the GPU executes many kernel instances in parallel

Triton sits between two worlds:

- higher-level tensor programming like PyTorch
- lower-level GPU programming like CUDA

It is lower level than PyTorch because you think about blocks, indexing, loads, stores, masks, and memory movement.

It is higher level than CUDA because Triton lets you express blocked tensor programs directly, and the compiler handles many scheduling and memory details.

## 2. Triton's core programming model

The official Triton guide describes its model as a blocked SPMD style.

That phrase matters.

- `SPMD` means many instances of the same program run in parallel on different pieces of data
- `blocked` means each program instance usually works on a block or tile, not just one scalar element

This is one of Triton's biggest differences from plain CUDA-style thinking.

Instead of thinking:

- one thread computes one scalar

you often think:

- one Triton program instance computes one tile of data

For example:

- vector add: one program handles one chunk of a vector
- softmax: one program may handle one row or several rows
- matmul: one program handles one tile of the output matrix

That blocked style is why Triton code often feels close to the math you want.

## 3. The smallest useful Triton mental model

When you read a Triton kernel, ask these five questions:

1. Which program instance am I?
2. Which indices does this instance own?
3. What values does it load?
4. What math does it do?
5. Where does it store the result?

If you can answer those five questions, you can usually understand the kernel.

## 4. The basic Triton pieces

### `@triton.jit`

This marks a function as a Triton kernel or Triton device function.

It is not executed as ordinary Python. Triton compiles it for the GPU.

### `tl.program_id(axis=...)`

This tells a running program instance which block of work it owns.

In a 1D launch:

- `tl.program_id(axis=0)` is the block id along the x dimension

In a 2D problem, you may conceptually map work along multiple axes.

### `tl.arange(...)`

This creates a vector of offsets inside the current block.

In vector add, a common pattern is:

```python
pid = tl.program_id(axis=0)
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

That means: "this program owns these `BLOCK_SIZE` positions."

### `tl.load(...)` and `tl.store(...)`

These are explicit memory reads and writes.

This is a major difference from regular PyTorch. In Triton you often directly express:

- what addresses to read
- what addresses to write
- what mask protects those accesses

### masks

Masks are how you safely handle edge cases.

Example:

```python
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

The mask says: only valid lanes should read real data. Invalid lanes get a fallback value.

This matters whenever your tensor size is not a perfect multiple of your tile size.

### `tl.constexpr`

These are compile-time constants, not normal runtime values.

Common examples:

- `BLOCK_SIZE`
- `BLOCK_SIZE_M`
- `BLOCK_SIZE_N`
- `num_stages`
- some tuning parameters

They matter because the compiler can optimize much more aggressively when those values are known at compile time.

## 5. How kernel launch works

A Triton kernel is launched with a grid, similar in spirit to CUDA.

Example:

```python
grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
kernel[grid](...)
```

Read that as:

- figure out how many blocks of work are needed
- launch one Triton program instance per block

So if you have:

- `n_elements = 10000`
- `BLOCK_SIZE = 1024`

then you launch enough programs to cover all 10000 elements.

This is one of the most important beginner ideas:

- the grid controls how many program instances exist
- the kernel code controls what each instance does

## 6. GPU architecture from the Triton point of view

From a Triton perspective, you do not need every hardware detail. You need the parts that affect kernel design.

### 6.1 Streaming multiprocessors or compute units

A GPU has many execution units that can run lots of work in parallel.

On NVIDIA, these are usually discussed as SMs.

From the Triton perspective, the important point is:

- many program instances are distributed across these multiprocessors
- your kernel should expose enough parallel work to keep them busy

If your grid is too small, the GPU is underutilized.

### 6.2 Warps or wavefronts

Hardware executes groups of lanes together.

On NVIDIA, the common unit is the warp.

From Triton's perspective, this matters because:

- control flow divergence can hurt performance
- memory access patterns across lanes matter a lot
- tuning parameters like `num_warps` change how much execution resource a program instance gets

You do not directly write warp-level code in the same way as CUDA intrinsics all the time, but the hardware still runs that way under the hood.

### 6.3 Registers

Registers are the fastest storage, private to active execution lanes.

From Triton's view:

- values you keep live in a kernel consume registers
- too much register pressure can reduce occupancy
- fused kernels can be great, but if they become too large they may use too many registers

This is one reason why "bigger fused kernel" is not always automatically better.

### 6.4 Shared memory or on-chip SRAM

The Triton docs often talk about keeping data on-chip or in SRAM.

The main practical idea is:

- on-chip storage is much faster than going back to DRAM
- fast kernels try to load data once, do lots of work on-chip, then store results back

This is exactly why the fused softmax tutorial is important.

Naive softmax reads and writes too much global memory. A fused Triton kernel reduces memory traffic by doing more work before writing back.

### 6.5 DRAM or global memory

This is the large off-chip memory where tensors live.

From a Triton point of view:

- global memory is expensive compared with registers and on-chip SRAM
- many kernels are memory-bandwidth bound, not math bound
- reducing memory traffic is often the main optimization lever

This explains why Triton is especially good for:

- fused elementwise ops
- softmax
- layer norm
- attention components

In these cases, avoiding extra reads and writes can matter more than raw arithmetic cost.

### 6.6 L2 cache

L2 sits between DRAM and the compute units.

You usually do not program L2 directly, but access order matters.

This shows up clearly in Triton's matmul tutorial:

- the order of tile execution affects L2 reuse
- grouped launch ordering can improve cache hit rate
- better data reuse can improve throughput significantly

So in Triton, even launch order can be a performance optimization.

### 6.7 Tensor cores and specialized units

Modern GPUs have specialized math units, such as tensor cores.

From Triton's perspective:

- some ops like `tl.dot` can map well to specialized hardware
- datatype choices matter a lot
- block shapes and compile-time parameters affect how well the compiler can use those units

The Triton programming guide explicitly mentions tensor-core-aware instruction selection as one of the optimizations the compiler can perform.

## 7. The real performance model: data movement first, math second

Many beginners think GPU optimization mostly means "do more FLOPs fast."

In Triton, a better starting view is:

- load less from DRAM
- reuse data on-chip
- write back less often
- keep enough parallel work in flight

That is why these ideas keep appearing:

- tiling
- fusion
- masking
- cache-friendly ordering
- software pipelining

The math is important, but memory movement often dominates performance.

## 8. Why tiling is central in Triton

Tiling means splitting a large problem into smaller blocks that fit the GPU better.

Examples:

- vector add: tile a long vector into chunks
- softmax: tile rows into blocks
- matmul: tile the output matrix into `BLOCK_SIZE_M x BLOCK_SIZE_N` blocks and reduce over `BLOCK_SIZE_K`

Why tiling helps:

- improves locality
- improves reuse of loaded data
- matches the blocked execution model of Triton
- makes it easier for the compiler to optimize scheduling and memory movement

Tiling is not just an optimization trick in Triton. It is basically the native way of thinking.

## 9. A simple reading of the three classic tutorials

### Vector add

What you learn:

- `program_id`
- `arange`
- load/store
- masks
- launch grid

This is the "hello world" of Triton.

### Fused softmax

What you learn:

- why fusion matters
- reductions like `tl.max` and `tl.sum`
- why SRAM/on-chip reuse beats repeated DRAM traffic
- why shape constraints and power-of-two block sizing show up

This is where Triton starts becoming practically interesting.

### Matrix multiplication

What you learn:

- multi-dimensional pointer arithmetic
- tiled compute
- accumulation in higher precision
- execution ordering for L2 reuse
- autotuning of block sizes and launch parameters

This is where Triton starts to feel like serious GPU programming.

## 10. Pointer arithmetic in Triton

This is the step from beginner to intermediate.

For a 2D tensor, memory addresses depend on strides.

That is why Triton matmul kernels pass things like:

- matrix sizes: `M`, `N`, `K`
- strides: `stride_am`, `stride_ak`, `stride_bk`, `stride_bn`, ...

The point is not to memorize formulas immediately. The point is to understand that:

- Triton kernels often compute blocks of addresses, not just blocks of values
- you build pointer grids, then load from them

This is a big mindset shift from PyTorch.

In PyTorch you write:

```python
c = a @ b
```

In Triton you think:

- which output tile does this program own?
- which A tile and B tile are needed?
- what pointers describe those tiles?
- how do we advance them along K?

## 11. Reductions and intermediate values

Once you move past elementwise kernels, Triton often becomes about reductions.

Examples:

- softmax needs max and sum reductions
- matmul reduces over the K dimension
- layer norm uses mean and variance-like reductions

The important learning step is:

- understand how a block loads data
- understand along which axis it reduces
- understand where the partial or final results live before storing

This is also where datatype decisions start to matter more.

For example, matmul often accumulates in fp32 even if inputs are fp16.

## 12. Occupancy, resource use, and trade-offs

A kernel does not run in a vacuum. It competes for hardware resources.

The main resources you think about in Triton are:

- registers
- shared memory / SRAM
- warps per program instance
- number of active program instances per multiprocessor

This leads to trade-offs:

- larger tiles may improve reuse, but consume more registers or SRAM
- more warps may improve throughput, but also raise resource usage
- more fusion may reduce memory traffic, but can increase register pressure

The fused softmax tutorial even computes occupancy-related values based on device properties and register/shared-memory usage. That is a strong hint that performance is not just algorithmic; it is also resource-shaped.

## 13. Autotuning

Autotuning is how Triton tries multiple configurations and keeps the best one.

Typical parameters to tune:

- `BLOCK_SIZE_M`
- `BLOCK_SIZE_N`
- `BLOCK_SIZE_K`
- `GROUP_SIZE_M`
- `num_warps`
- `num_stages`

Why this matters:

- different GPUs like different tile shapes
- different problem sizes like different configurations
- the best settings are hard to predict perfectly by hand

Autotuning is one of the bridges from "correct Triton kernel" to "fast Triton kernel."

## 14. What Triton's compiler helps with

The Triton programming guide emphasizes that the compiler can do a lot of heavy lifting, including:

- coalescing
- vectorization
- prefetching
- shared-memory allocation and synchronization
- asynchronous copy scheduling
- tensor-core-aware instruction selection

This is why Triton is attractive.

You still think carefully about tiles and memory access, but you do not manually orchestrate every low-level detail the way you might in raw CUDA.

## 15. Warp specialization, from a practical perspective

The PyTorch blog on warp specialization describes a newer direction in Triton: letting different warps inside a kernel specialize in different roles.

Why do this?

- reduce control-flow divergence
- overlap memory and compute better
- use specialized hardware units more effectively

The practical mental model is:

- in a simple kernel, all warps may do roughly the same kind of work
- in a more advanced kernel, some warps can focus on producing data, some on compute, some on epilogues or communication

This is more advanced than what you need at the start, but it tells you something important:

- high-end Triton performance increasingly depends on smarter scheduling across hardware structure, not just writing the correct math

## 16. Common beginner mistakes

### Thinking Triton replaces PyTorch

It does not. Usually Triton complements PyTorch.

### Focusing on syntax before indexing

Most Triton bugs are really indexing, shape, stride, or masking bugs.

### Ignoring correctness while chasing performance

Always compare against PyTorch first.

### Writing giant kernels too early

Start small. Elementwise kernels and softmax teach a lot.

### Ignoring memory traffic

Many slow kernels are slow because they move data badly, not because the arithmetic is wrong.

## 17. What you should learn in order

If you want a solid path from basics to medium level, learn in this order:

1. launch grid and `program_id`
2. `tl.arange`, indexing, and masks
3. `tl.load` and `tl.store`
4. simple elementwise kernels
5. reductions like max and sum
6. row-wise kernels like softmax
7. multi-dimensional pointer arithmetic
8. tiled matmul
9. autotuning and performance measurement
10. advanced scheduling ideas like warp specialization

## 18. A good working definition of "intermediate Triton"

You are around intermediate level when you can:

- write a correct elementwise kernel without copying from a tutorial
- explain how a program instance maps to a block of output
- use masks correctly for edge handling
- reason about strides and 2D pointer arithmetic
- write or understand a simple reduction kernel
- explain why a kernel is memory-bound or compute-bound
- tune block sizes and compare variants fairly

## 19. The shortest architecture summary from Triton's point of view

Here is the compact view:

- tensors live in global memory
- the GPU has many multiprocessors executing groups of lanes together
- on-chip storage and registers are precious and fast
- memory traffic is expensive
- tiling improves reuse and locality
- launch order can affect cache reuse
- tensor cores and specialized units matter for dense math
- Triton's job is to let you describe blocked tensor programs while the compiler maps them efficiently to hardware

## 20. What to read next

In this repo, read these next:

- `mental_model.md`
- `learning.md`
- `main.py`

Then study the official Triton materials in this order:

1. vector add tutorial
2. fused softmax tutorial
3. matrix multiplication tutorial
4. Triton programming guide introduction

## 21. One-sentence takeaway

Triton is easiest to understand when you think of it as blocked GPU programming for tensor problems, where performance mostly comes from good tiling, good memory movement, and enough parallel work to keep the hardware busy.
