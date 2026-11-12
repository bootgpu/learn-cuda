#include <cuda_runtime.h>

// Each thread computes two elements of c = a + b.
__global__ void vecAddKernel(const float* a, const float* b, float* c, int n) {
    // TODO
}

// solve() is supplied by the judge; do not write your own. It runs
//
//     int threads = 256;
//     int blocks = (n + 2 * threads - 1) / (2 * threads);
//     vecAddKernel<<<blocks, threads>>>(a, b, c, n);
