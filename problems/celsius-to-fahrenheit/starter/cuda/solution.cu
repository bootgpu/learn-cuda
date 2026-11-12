#include <cuda_runtime.h>

__global__ void celsiusToFahrenheitKernel(const float* celsius, float* fahrenheit, int n) {
    // TODO: one thread per element: fahrenheit = celsius * 9 / 5 + 32
}

// celsius, fahrenheit are device pointers
extern "C" void solve(const float* celsius, float* fahrenheit, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    celsiusToFahrenheitKernel<<<blocks, threads>>>(celsius, fahrenheit, n);
}
