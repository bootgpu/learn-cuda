#include <cuda_runtime.h>

__global__ void map_kernel(const float* x, float* y, int N) {
    // TODO: y[i] = 2 * x[i] + 1 for this thread's i
}

// x, y are device pointers
extern "C" void solve(const float* x, float* y, int N) {
    // TODO: launch map_kernel
}
