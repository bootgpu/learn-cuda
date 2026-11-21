#include "solution.cu"

// Fixed for this stage: one-dimensional, 256 threads per block, one thread per student.
extern "C" void solve(const float* scores, const float* weights, float* totals, int students,
                      int subjects) {
    int threads = 256;
    int blocks = (students + threads - 1) / threads;
    weightedScoresKernel<<<blocks, threads>>>(scores, weights, totals, students, subjects);
}
