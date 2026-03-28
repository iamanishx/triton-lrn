# CUDA Architecture Complete Guide

> Everything you need to know about CUDA thread hierarchy, memory, and performance

---

## Table of Contents

1. [GPU Architecture Overview](#gpu-overview)
2. [Thread Hierarchy](#thread-hierarchy)
3. [Grid and Block](#grid-and-block)
4. [Warp Execution](#warp)
5. [SM (Streaming Multiprocessor)](#sm)
6. [Memory Hierarchy](#memory-hierarchy)
7. [Shared Memory & Tiling](#shared-memory)
8. [Memory Coalescing](#coalescing)
9. [Launch Configuration](#launch)

---

<a name="gpu-overview"></a>

## 1. GPU Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          GPU                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                     GRID (entire workload)                    │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│         ┌────────────────────┼────────────────────┐               │
│         │                    │                    │               │
│         ▼                    ▼                    ▼               │
│   ┌───────────┐       ┌───────────┐       ┌───────────┐         │
│   │  BLOCK 0  │       │  BLOCK 1  │       │  BLOCK 2  │   ...   │
│   │  (on SM 0)│       │  (on SM 1)│       │  (on SM 0)│         │
│   └───────────┘       └───────────┘       └───────────┘         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### GPU vs CPU

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Cores** | Few (4-32) | Thousands |
| **Focus** | Low latency | High throughput |
| **Cache** | Large (MBs) | Small (KBs) |
| **Threads** | Few, fast | Many, simple |
| **Control** | Complex (branch prediction) | Simple (SIMT) |

---

<a name="thread-hierarchy"></a>

## 2. Thread Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COMPLETE THREAD HIERARCHY                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                         GRID                                  │  │
│   │  - Contains ALL blocks                                        │  │
│   │  - Can be 1D, 2D, or 3D                                       │  │
│   │  - Defined by gridDim (gridDim.x, gridDim.y, gridDim.z)     │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│   ┌──────────────────────────┼──────────────────────────┐          │
│   │                          │                          │          │
│   ▼                          ▼                          ▼          │
│ ┌────────┐              ┌────────┐              ┌────────┐        │
│ │ BLOCK 0│              │ BLOCK 1│              │ BLOCK 2│        │
│ │        │              │        │              │        │        │
│ │ Threads│              │ Threads│              │ Threads│        │
│ │   0-255│              │   0-255│              │   0-255│        │
│ │        │              │        │              │        │        │
│ │ - 256 threads         │ - 256 threads        │ - 256 threads   │
│ │ - Runs on ONE SM      │ - Runs on ONE SM     │ - Runs on ONE SM│
│ │ - Can sync            │ - Can sync           │ - Can sync      │
│ │ - Shared memory       │ - Shared memory      │ - Shared memory │
│ └────────┘              └────────┘              └────────┘        │
│      │                        │                        │          │
│      ▼                        ▼                        ▼          │
│ ┌────────┐              ┌────────┐              ┌────────┐        │
│ │ WARP 0 │              │ WARP 0 │              │ WARP 0 │        │
│ │ T0-T31 │              │ T0-T31 │              │ T0-T31 │        │
│ ├────────┤              ├────────┤              ├────────┤        │
│ │ WARP 1 │              │ WARP 1 │              │ WARP 1 │        │
│ │T32-T63 │              │T32-T63 │              │T32-T63 │        │
│ └────────┘              └────────┘              └────────┘        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

<a name="grid-and-block"></a>

## 3. Grid and Block

### What is a Grid?

- **Grid** = Collection of ALL blocks
- Represents the entire problem
- Can be 1D, 2D, or 3D

### What is a Block?

- **Block** = Group of threads
- Runs on ONE Streaming Multiprocessor (SM)
- Maximum **1024 threads** per block
- Threads in a block can:
  - Synchronize with `__syncthreads()`
  - Share data via `__shared__` memory
  - Communicate directly

### Dimensions

```cuda
// 1D grid, 1D block
dim3 gridSize(4);      // 4 blocks
dim3 blockSize(256);   // 256 threads per block
// Total: 4 × 256 = 1024 threads

// 2D grid, 2D block (common for matrices)
dim3 gridSize(4, 4);      // 16 blocks total
dim3 blockSize(16, 16);   // 256 threads per block
// Total: 16 × 256 = 4096 threads

// 3D (for volumes)
dim3 gridSize(2, 2, 2);      // 8 blocks
dim3 blockSize(8, 8, 8);     // 512 threads per block
// Total: 8 × 512 = 4096 threads
```

### Thread ID Calculation

```cuda
__global__ void printThreadIds() {
    // Thread within its block
    int tid_in_block = threadIdx.x 
                    + threadIdx.y * blockDim.x 
                    + threadIdx.z * blockDim.x * blockDim.y;
    
    // Block ID
    int bid = blockIdx.x 
            + blockIdx.y * gridDim.x 
            + blockIdx.z * gridDim.x * gridDim.y;
    
    // Total threads per block
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    
    // Global unique ID
    int global_id = bid * threads_per_block + tid_in_block;
    
    printf("Block:(%d,%d,%d) Thread:(%d,%d,%d) GlobalID:%d\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z,
           global_id);
}

// Launch with 2x2 blocks, 4x4 threads each
printThreadIds<<<dim3(2,2), dim3(4,4)>>>();
// 4 blocks × 16 threads = 64 total threads
```

---

<a name="warp"></a>

## 4. Warp Execution

### What is a Warp?

- **Warp** = Group of **32 threads**
- ALL threads in a warp execute the **same instruction** at the same time
- This is why GPUs are so fast - SIMT (Single Instruction Multiple Threads)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WARP EXECUTION                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   WARP = 32 threads executing IN LOCKSTEP                            │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  ┌──┬──┬──┬──┬──┬──┬──┬──┬────┬──┬──┐                       │   │
│   │  │T0│T1│T2│T3│T4│T5│...│T30│T31│                        │   │
│   │  └──┴──┴──┴──┴──┴──┴──┴────┴──┴──┘                       │   │
│   │    │  │  │  │  │  │  │    │   │                          │   │
│   │    ▼  ▼  ▼  ▼  ▼  ▼  ▼    ▼   ▼                          │   │
│   │   ┌─────────────────────────────┐                         │   │
│   │   │  SAME INSTRUCTION (ADD, MUL)│                         │   │
│   │   └─────────────────────────────┘                         │   │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                      │
│   IF (condition) {                                                  │
│       // Threads diverge! Half take branch A, half take branch B   │
│   }                                                                  │
│                                                                      │
│   ⚠️  DIVERGENCE = PERFORMANCE PENALTY!                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Warp Divergence

```cuda
// BAD - causes divergence
if (threadIdx.x % 2 == 0) {
    // Half of warp executes this
} else {
    // Other half executes this
}

// GOOD - no divergence
if (threadIdx.x < 16) {
    // First half of warp (threads 0-15)
} else {
    // Second half (threads 16-31)
}
```

### Best Practices

| Rule | Why |
|------|-----|
| Keep block size a multiple of 32 | Full warps = efficiency |
| Avoid branch divergence | Half-warp penalty |
| Threads in warp should follow same path | Same instruction |

---

<a name="sm"></a>

## 5. SM (Streaming Multiprocessor)

### What is an SM?

- **SM** = Streaming Multiprocessor
- The actual processor on the GPU
- Executes blocks
- Each GPU has multiple SMs

```
┌─────────────────────────────────────────────────────────────────────┐
│                   STREAMING MULTIPROCESSOR (SM)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                         SM                                   │   │
│   │                                                                  │  │
│   │  ┌─────────────────────────────────────────────────────┐   │   │
│   │  │              Warp Schedulers (4)                      │   │   │
│   │  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐      │   │   │
│   │  │  │ Warp 0 │  │ Warp 1 │  │ Warp 2 │  │ Warp 3 │      │   │   │
│   │  │  └────────┘  └────────┘  └────────┘  └────────┘      │   │   │
│   │  └─────────────────────────────────────────────────────┘   │   │
│   │                           │                                   │   │
│   │  ┌─────────────────────────────────────────────────────┐   │   │
│   │  │              Execution Units                          │   │   │
│   │  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     │   │   │
│   │  │  │ FP32 │ │ FP64 │ │ INT32│ │ LD/ST│ │ SFU  │     │   │   │
│   │  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘     │   │   │
│   │  └─────────────────────────────────────────────────────┘   │   │
│   │                           │                                   │   │
│   │  ┌─────────────────────────────────────────────────────┐   │   │
│   │  │              128 KB Shared Memory / L1 Cache        │   │   │
│   │  └─────────────────────────────────────────────────────┘   │   │
│   │  ┌─────────────────────────────────────────────────────┐   │   │
│   │  │              64 KB Registers (16,384 registers)    │   │   │
│   │  └─────────────────────────────────────────────────────┘   │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   Can run: 16-32 blocks, hundreds of threads simultaneously         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### SM Resources

| Resource | Amount | Shared By |
|----------|--------|-----------|
| **Shared Memory** | 128 KB | All blocks on SM |
| **Registers** | 64 KB | All threads |
| **L1 Cache** | 128 KB | All threads |
| **Constant Cache** | 64 KB | All threads |
| **Max Blocks** | 16-32 | Per SM |
| **Max Warps** | 64 | Per SM |

---

<a name="memory-hierarchy"></a>

## 6. Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                       MEMORY HIERARCHY                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   DEVICE (GPU)                                                       │
│   ┌───────────────────────────────────────────────────────────────┐ │
│   │                     GLOBAL MEMORY (VRAM)                       │ │
│   │  - 8-24 GB (HBM/GDDR6)                                        │ │
│   │  - Latency: ~500 memory cycles                                 │ │
│   │  - All threads can read/write                                 │ │
│   │  - cudaMalloc(), cudaMemcpy()                                │ │
│   │  - Persists between kernels                                   │ │
│   └───────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│          ┌──────────────────┼──────────────────┐                   │
│          │                  │                  │                   │
│          ▼                  ▼                  ▼                   │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │
│   │     SM 0    │    │     SM 1    │    │     SM N    │           │
│   │ ┌─────────┐ │    │ ┌─────────┐ │    │ ┌─────────┐ │           │
│   │ │Shared   │ │    │ │Shared   │ │    │ │Shared   │ │           │
│   │ │Memory   │ │    │ │Memory   │ │    │ │Memory   │ │           │
│   │ │ 128 KB  │ │    │ │ 128 KB  │ │    │ │ 128 KB  │ │           │
│   │ └─────────┘ │    │ └─────────┘ │    │ └─────────┘ │           │
│   │ ┌─────────┐ │    │ ┌─────────┐ │    │ ┌─────────┐ │           │
│   │ │  L1    │ │    │ │  L1    │ │    │ │  L1    │ │           │
│   │ │Cache   │ │    │ │Cache   │ │    │ │Cache   │ │           │
│   │ │ 128 KB │ │    │ │ 128 KB │ │    │ │ 128 KB │ │           │
│   │ └─────────┘ │    │ └─────────┘ │    │ └─────────┘ │           │
│   └─────────────┘    └─────────────┘    └─────────────┘           │
│          │                  │                  │                   │
│          ▼                  ▼                  ▼                   │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │
│   │  Registers │    │  Registers │    │  Registers │           │
│   │  64 KB/SM  │    │  64 KB/SM  │    │  64 KB/SM  │           │
│   │ 255/thread │    │ 255/thread │    │ 255/thread │           │
│   └─────────────┘    └─────────────┘    └─────────────┘           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Memory Comparison

| Memory | Location | Size | Latency | Scope |
|--------|----------|------|---------|-------|
| **Registers** | On-chip | 64 KB/SM |  |
| **Shared0 cycles | Thread Memory** | On-chip | 128 KB/SM | ~4 cycles | Block |
| **L1 Cache** | On-chip | 128 KB/SM | ~4 cycles | Thread |
| **Global Memory** | Off-chip | 8-24 GB | ~500 cycles | All |
| **Host Memory** | CPU RAM | GBs | ~1000 cycles | CPU-GPU |

### Using Different Memory Types

```cuda
__global__ void memoryExample(float* global_data) {
    // GLOBAL MEMORY - slow, persistent
    float value = global_data[threadIdx.x];
    
    // SHARED MEMORY - fast, per-block
    __shared__ float shared_data[256];
    shared_data[threadIdx.x] = value;
    __syncthreads();
    
    // REGISTERS - fastest, per-thread
    float register_value = value * 2.0f;
}
```

---

<a name="shared-memory"></a>

## 7. Shared Memory & Tiling

### What is Shared Memory?

- **On-chip memory** (very fast)
- **48 KB** per block (configurable with L1)
- Shared by all threads in a block
- Used for caching frequently accessed data

### Tiling Optimization

**Problem:** Global memory is slow (500+ cycles)

**Solution:** Use shared memory as a cache

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TILE MATRIX MULTIPLICATION                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Naive approach (slow):                                            │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Each thread loads from slow global memory for EVERY        │   │
│   │  element in the multiplication                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   Tiled approach (fast):                                            │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  1. Load tile into fast shared memory                       │   │
│   │  2. All threads in block use shared memory                 │   │
│   │  3. Process tile                                            │   │
│   │  4. Repeat for next tile                                    │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Tiled Matrix Multiply Example

```cuda
#define TILE_SIZE 16

__global__ void tiledMatrixMul(float* A, float* B, float* C, int N) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Thread's position in output
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Accumulate result for this element
    float sum = 0.0f;
    
    // Number of tiles needed
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < numTiles; tile++) {
        // Load tile from global to shared memory
        // Thread (i,j) loads A[row, tile*TS+j] and B[tile*TS+i, col]
        int aCol = tile * TILE_SIZE + threadIdx.x;
        int bRow = tile * TILE_SIZE + threadIdx.y;
        
        if (row < N && aCol < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && bRow < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        // Wait for all threads to load
        __syncthreads();
        
        // Compute partial result from this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        // Wait before loading next tile
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Launch: each block handles TILE_SIZE x TILE_SIZE output elements
dim3 blockSize(TILE_SIZE, TILE_SIZE);
dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, 
              (N + TILE_SIZE - 1) / TILE_SIZE);
tiledMatrixMul<<<gridSize, blockSize>>>(A, B, C, N);
```

### Why Tiling Works

| Aspect | Naive | Tiled |
|--------|-------|-------|
| Global memory accesses | N³ | N³/TILE_SIZE |
| Shared memory accesses | 0 | N³ |
| Performance | Slow | 2-4x faster |

---

<a name="coalescing"></a>

## 8. Memory Coalescing

### What is Coalescing?

- **Coalesced memory access** = adjacent threads access adjacent memory addresses
- Maximizes memory bandwidth
- Critical for performance!

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COALESCED vs NON-COALESCED                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   COALESCED (GOOD):                                                  │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  Thread 0 ──► Address 100                                   │  │
│   │  Thread 1 ──► Address 101  (adjacent!)                      │  │
│   │  Thread 2 ──► Address 102  (adjacent!)                      │  │
│   │  Thread 3 ──► Address 103  (adjacent!)                      │  │
│   │  ...                                                          │  │
│   │  Memory transactions: 1 (all in one 128-byte line)         │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                      │
│   NON-COALESCED (BAD):                                              │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  Thread 0 ──► Address 100                                   │  │
│   │  Thread 1 ──► Address 200  (scattered!)                    │  │
│   │  Thread 2 ──► Address 300  (scattered!)                    │  │
│   │  Thread 3 ──► Address 400  (scattered!)                    │  │
│   │  ...                                                          │  │
│   │  Memory Transactions: 32+ (one per thread!)                 │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Coalesced Access Patterns

```cuda
// COALESCED (GOOD) - row-major order
__global__ void coalescedAccess(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[idx];  // Thread i reads address i
}

// NON-COALESCED (BAD) - column-major access
__global__ void nonCoalescedAccess(float* data, int N) {
    int idx = threadIdx.x * N;  // Thread 0 reads 0, N, 2N...
    float value = data[idx];
}

// COALESCED - Matrix row access
__global__ void matrixRowAccess(float* matrix, int N) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    float value = matrix[row * N + col];  // Adjacent for threadIdx
}

// NON-COALESCED - Matrix column access  
__global__ void matrixColAccess(float* matrix, int N) {
    int row = threadIdx.x;
    int col = blockIdx.x;
    float value = matrix[row * N + col];  // Scattered!
}
```

### Row-Major vs Column-Major

```
Row-Major (C/C++, row by row):
┌─────────────────────────────┐
│ [0,0] [0,1] [0,2] [0,3]   │  ← Elements 0-3 adjacent in memory
│ [1,0] [1,1] [1,2] [1,3]   │  ← Elements 4-7 adjacent
│ [2,0] [2,1] [2,2] [2,3]   │
└─────────────────────────────┘

Access pattern: [i][j] = row * N + col
```

### Tips for Coalescing

| Do | Don't |
|----|-------|
| Access global memory sequentially | Random access patterns |
| Align data to 128-byte boundaries | Unaligned access |
| Use row-major for C++ | Column-major in row-major code |
| Read contiguous chunks | Strided access |

---

<a name="launch"></a>

## 9. Launch Configuration

### Syntax

```cuda
kernel_name<<<grid, block, shared_mem, stream>>>(args);
```

| Parameter | Description |
|-----------|-------------|
| `grid` | dim3 - number of blocks |
| `block` | dim3 - threads per block |
| `shared_mem` | Shared memory per block (bytes) |
| `stream` | Stream ID (default 0) |

### Common Configurations

```cuda
// Vector addition - 1D
int N = 1000000;
int threads = 256;
int blocks = (N + threads - 1) / threads;
vectorAdd<<<blocks, threads>>>(A, B, C, N);

// Matrix operation - 2D
dim3 blockSize(16, 16);
dim3 gridSize((N + 15) / 16, (M + 15) / 16);
matrixMul<<<gridSize, blockSize>>>(A, B, C, M, N, K);

// 3D volume processing
dim3 blockSize(8, 8, 8);
dim3 gridSize((X+7)/8, (Y+7)/8, (Z+7)/8);
volumeProcess<<<gridSize, blockSize>>>(volume, size);
```

### Choosing Block Size

| Factor | Recommendation |
|--------|----------------|
| **General** | 128-256 threads |
| **Memory bound** | 256 threads |
| **Compute bound** | 128 threads |
| **Shared memory bound** | 64-128 threads |
| **Must be multiple of 32** | Warp size! |

### Occupancy

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OCCUPANCY                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Occupancy = (Active Warps) / (Max Warps per SM)                  │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  100% Occupancy                                             │   │
│   │  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐  │   │
│   │  │Warp│ │Warp│ │Warp│ │Warp│ │Warp│ │Warp│ │Warp│ │Warp│  │   │
│   │  └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘  │   │
│   │   Active warps hiding memory latency                        │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   Low occupancy = more memory stall waiting                         │
│                                                                      │
│   Formula:                                                          │
│   blocks_per_sm = min(max_blocks, registers / regs_per_block)      │
│   threads_per_block = blocks_per_sm * threads_per_block           │
│   occupancy = threads_per_block / max_threads_per_sm              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

| Concept | Key Points |
|---------|------------|
| **Grid** | All blocks, 1D/2D/3D |
| **Block** | Max 1024 threads, runs on one SM |
| **Warp** | 32 threads, execute together |
| **SM** | Executes blocks, has shared memory |
| **Global Memory** | Slow, 8-24 GB |
| **Shared Memory** | Fast, 48 KB/block |
| **Registers** | Fastest, 255/thread |
| **Coalescing** | Adjacent threads → adjacent memory |
| **Tiling** | Use shared mem as cache |
| **Divergence** | Avoid if branches in warp |

---

## What's Next?

1. **cuBLAS** - Optimized matrix operations
2. **cuDNN** - Deep learning primitives
3. **TensorRT** - Inference optimization
4. **Streams** - Concurrent kernel execution
5. **Unified Memory** - Simpler memory management
