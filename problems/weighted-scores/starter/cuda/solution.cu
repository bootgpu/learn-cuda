#include <cuda_runtime.h>

__global__ void weightedScoresKernel(const float* scores, const float* weights, float* totals,
                                     int students, int subjects) {
    // TODO: one thread per student
}

// solve() is supplied by the judge (see harness.cu); do not write your own.
