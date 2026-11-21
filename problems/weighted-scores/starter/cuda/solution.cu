#include <cuda_runtime.h>

__global__ void weightedScoresKernel(const float* scores, const float* weights, float* totals,
                                     int students, int subjects) {
    // TODO: one thread per student
}

// scores, weights, totals are device pointers
extern "C" void solve(const float* scores, const float* weights, float* totals, int students,
                      int subjects) {
    int threads = 256;
    int blocks = (students + threads - 1) / threads;
    weightedScoresKernel<<<blocks, threads>>>(scores, weights, totals, students, subjects);
}
