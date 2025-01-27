# CUDA 
-------------------------------------------------------------------------------
### Overview for addition program:
This program demonstrates a basic CUDA implementation for vector addition. It uses the GPU to parallelize computations,
---
Day 1:
### Theoretical Learnings : 
- CUDA requires explicit memory management between the host (CPU) and the device (GPU).
- Threads and blocks enable parallel processing by dividing data across the GPU.
- Kernel functions (`__global__`) execute on the GPU and must be launched with specific thread and block configurations ( as per sytem limits).
- PTX (Parallel Thread Execution) code is generated during compilation and later JIT (just In Time) compiled into machine code by the CUDA driver.
- Machine code for the kernel is stored in the GPU's instruction cache for execution.
---------------------------------------------------------------------------------


