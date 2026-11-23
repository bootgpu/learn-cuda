#include "solution.cu"

// Fixed for this stage: one-dimensional, 32 threads per block, one thread per row of P.
extern "C" void solve(const float* M, const float* N, float* P, int n) {
    int threads = 32;
    int blocks = (n + threads - 1) / threads;
    matMulRowKernel<<<blocks, threads>>>(M, N, P, n);
}
