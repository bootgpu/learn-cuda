#include <cuda_runtime.h>

// One thread per pixel: x is the column, y the row, z the frame.
__global__ void videoFadeKernel(const float* input, float* output, int frames, int height, int width) {
    // TODO
}

// solve() is supplied by the judge (see harness.cu); do not write your own.
