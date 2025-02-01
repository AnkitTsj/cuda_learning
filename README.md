# CUDA 
-------------------------------------------------------------------------------
### Overview for addition program:
####This program demonstrates a basic CUDA implementation for vector addition. It uses the GPU to parallelize computations,
---
Day 1:
### Theoretical Learnings : 
- CUDA requires explicit memory management between the host (CPU) and the device (GPU).
- Threads and blocks enable parallel processing by dividing data across the GPU.
- Kernel functions (`__global__`) execute on the GPU and must be launched with specific thread and block configurations ( as per sytem limits).
- PTX (Parallel Thread Execution) code is generated during compilation and later JIT (just In Time) compiled into machine code by the CUDA driver.
- Machine code for the kernel is stored in the GPU's instruction cache for execution.
---------------------------------------------------------------------------------

Day 5: Understanding Row and Column Indexing in 1D Memory 

## How Rows and Columns are Handled in 1D Memory
When working with CUDA, a **2D matrix is stored in a 1D memory layout**. To efficiently access elements, we need to compute the correct **row and column indices** in terms of memory addresses. This is done using **thread and block indices**.

### 1. **Row Calculation**
Each thread computes its **row index** using:
```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
```
- `blockIdx.y * blockDim.y` gives the **starting row index for a block ( we get 0 - 1 in our case) .**.
- `threadIdx.y` adds the **thread index within the block**, resulting in the actual **global row index (0 -31 in our case)**.

Thus, **all row indices** are generated correctly for **each thread across all blocks**.

### 2. **Converting 2D Indexing (and indices) to 1D Memory Address**
To get the **correct memory location** in 1D, we use:
```cpp
sharedA[threadIdx.y][threadIdx.x] = A[row * n + k * BLOCK_SIZE + threadIdx.x];
```
Breaking it down:
- `row * n`: Since **each row has `n` elements**, multiplying by `n` gives the **starting index of that row** in 1D memory.
- `k * BLOCK_SIZE`: This adds an **offset for the block-wise shift**, ensuring correct indexing within each tile.
- `threadIdx.x`: Finally, this **adds the column offset** for the specific thread.

Thus, the complete indexing formula ensures that each thread fetches the correct matrix element.

### 3. **Final Formula for Index Calculation**
For each thread, the index in memory is computed as:
```cpp
Index = row * n + k * BLOCK_SIZE + threadIdx.x
```
Where:
- `row * n` → Row start index
- `k * BLOCK_SIZE` → Block-wise shift for matrices (blocks/tile)
- `threadIdx.x` → Column offset for each thread

## Shared Memory Allocation Per Block 
In CUDA, **shared memory is allocated per block**, meaning:
- Each **block gets a dedicated shared memory region**.
- **All threads within the same block share the same shared memory**.
- The memory is **not shared between blocks**, meaning each block works with its own local copy.

For example, when declaring:
```cpp
__shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
```
- Each **block gets its own copy** of `sharedA`, independent of other blocks.
- **Threads within the block** can access `sharedA`, enabling fast intra-block communication.

## Summary
- **Row indices** are computed using `blockIdx.y * blockDim.y + threadIdx.y`.
- **1D memory indexing** is handled by `row * n + k * BLOCK_SIZE + threadIdx.x`.
- **Shared memory is allocated per block**, ensuring efficient memory access and avoiding redundant global memory accesses.
(since the internal parallelism is a stress for warps and threads, we are free!!)
This ensures efficient parallel computation on a **2D matrix using CUDA**. 


