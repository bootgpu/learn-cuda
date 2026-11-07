#include <cuda_runtime.h>

// x, y are device pointers
extern "C" void solve(const float* x, float* y, int N) {
    // TODO: find the maximum, then the sum of exp(x[i] - max), then write y
}
