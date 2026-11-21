#include "solution.cu"

// Fixed for this stage: 8 x 8 x 4 threads per block; x is the column, y the row, z the frame.
extern "C" void solve(const float* input, float* output, int frames, int height, int width) {
    dim3 block(8, 8, 4);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
              (frames + block.z - 1) / block.z);
    videoFadeKernel<<<grid, block>>>(input, output, frames, height, width);
}
