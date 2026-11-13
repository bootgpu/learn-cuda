This quiz covers these parts of chapter 2: data parallelism, thread indices, the rules for running blocks, managing and moving device memory from the host, error checking, and the function keywords.

While answering, you can refer to the complete host function for vector addition, which puts the five steps together:

```cpp
void vecAdd(const float* h_a, const float* h_b, float* h_c, int n) {
    size_t bytes = n * sizeof(float);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
```

Pointers starting with `h_` point to host memory; those starting with `d_` point to device memory.
