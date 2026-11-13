::bilibili[第 2 章视频：从循环到 kernel]{bvid="BV1nkW962EJS"}

本测验覆盖第 2 章的这些内容：数据并行、线程编号、block 的执行规则、host 端的显存管理与数据搬运，以及错误检查和函数关键字。

答题时可以对照视频里完整的 host 函数，它把向量加法要走的五步串在一起：

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

`h_` 开头的指针指向内存（host），`d_` 开头的指向显存（device）。
