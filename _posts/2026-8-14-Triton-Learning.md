---
layout: post
title: Triton Learning
tags: [GPU, triton, PL]
math: true
mermaid: true
---

This is an exercise record of problems from [xpuoj](https://xpuoj.com). For better understanding, I will attach CUDA code for several problems.

## Problem

### [1. a += b problem (fp16)](https://xpuoj.com/p/1)

- triton

```python
import triton
import triton.language as tl

@triton.jit # similar to __global__, indicate this is a triton kernel
def vector_add(
    A_ptr, # triton converts PyTorch tensor to GPU pointer
    B_ptr,
    N,
    B: tl.constexpr, # tl.constexpr means the value must be fixed during compilation
):
    # program == block in cuda
    pid = tl.program_id(axis=0)
    # triton focuses on program rather than thread/warp, it deal with a set of data in a program(block)
    # offsets is a vector!
    offsets = pid * B + tl.arange(0, B)

    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)

    tl.store(A_ptr + offsets, a + b, mask=mask)

def run_kernel(
    A,  # Tensor[fp16]
    B,  # Tensor[fp16]
    numel,  # int64
):
    # set block size (similar to thread in cuda, thread size <= 1024)
    BLOCK = 1024 
    A = A.contiguous() # make A continuous in memory
    B = B.contiguous() # make B continuous in memory
    grid = (triton.cdiv(numel, BLOCK), ) # calculate the 
    vector_add[grid](A, B, numel, BLOCK); # call triton kernel
    return A
```

- CUDA

```c
#include <stdint.h>
#include <cuda_fp16.h>

__global__ void vectorAdd(__half* A, const __half* B, int64_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[i] += B[i];
    }
}

extern "C" void run_kernel(__half* A, const __half* B, int64_t numel) {
    const int L = 1024;
    dim3 gridSize = (numel + L - 1) / L;
    dim3 blockSize = L;
    vectorAdd<<<gridSize, blockSize>>>(A, B, numel);
}
```
