#include <cuda_runtime.h>

// One thread per element.
__global__ void transposeKernel(const float* input, float* output, int rows, int cols) {
    // TODO
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 block(16, 16);
    dim3 grid(1, 1);  // TODO: enough blocks to cover every element
    transposeKernel<<<grid, block>>>(input, output, rows, cols);
}
