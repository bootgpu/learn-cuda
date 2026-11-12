#include "solution.cu"

// Fixed for this stage: 256 threads per block, one thread for every two elements.
extern "C" void solve(const float* a, const float* b, float* c, int n) {
    int threads = 256;
    int blocks = (n + 2 * threads - 1) / (2 * threads);
    vecAddKernel<<<blocks, threads>>>(a, b, c, n);
}
