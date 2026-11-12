#include <cuda_runtime.h>

// Each thread computes two elements of c = a + b.
__global__ void vecAddKernel(const float* a, const float* b, float* c, int n) {
    // TODO
}

// solve() is supplied by the judge (see harness.cu); do not write your own.
