#include <cuda_runtime.h>

__global__ void sum_kernel(const float* x, float* sum, int N) {
    // TODO: sum this block's part of x, then atomicAdd it to sum[0]
}

// x, sum are device pointers; sum[0] is undefined on entry
extern "C" void solve(const float* x, float* sum, int N) {
    // TODO: set sum[0] to 0, then launch sum_kernel
}
