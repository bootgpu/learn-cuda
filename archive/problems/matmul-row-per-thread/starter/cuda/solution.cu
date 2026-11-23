#include <cuda_runtime.h>

__global__ void matMulRowKernel(const float* M, const float* N, float* P, int n) {
    // TODO: one thread per row of P
}

// solve() is supplied by the judge (see harness.cu); do not write your own.
