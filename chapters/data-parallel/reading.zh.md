::bilibili[第 2 章视频：第一个 CUDA 程序]{bvid="BV1nkW962EJS"}

本页汇总视频中的代码与公式，做题时可以直接对照。最后两节是视频没有展开、做题时要用到的内容。

## kernel：全局编号与边界检查

```cpp
__global__ void vecAddKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 全局编号
    if (i < n) {                                    // 多出的线程不做任何事
        c[i] = a[i] + b[i];
    }
}
```

| 变量          | 含义                                   |
| ------------- | -------------------------------------- |
| `threadIdx.x` | 本线程在所属 block 中的编号，从 0 开始 |
| `blockIdx.x`  | 本线程所属 block 的编号，从 0 开始     |
| `blockDim.x`  | 每个 block 的线程数                    |

`blockIdx.x * blockDim.x` 是之前各 block 共占用的位置数，加上 `threadIdx.x`，即为本线程在整个 grid 中的位置。`n` 不一定是每个 block 线程数的整数倍，多出的线程由 `if (i < n)` 排除。

## 启动配置：向上取整

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;  // 向上取整
vecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
```

`<<<…>>>` 中第一个数是 block 的个数，第二个数是每个 block 的线程数。一个 block 最多 1024 个线程，常用 256。直接写 `n / threads` 会舍去小数部分，最后几个元素无人计算。

## 完整的 host 函数

```cpp
void vecAdd(const float* h_a, const float* h_b, float* h_c, int n) {
    int size = n * sizeof(float);
    float *d_a, *d_b, *d_c;

    // 1. 分配显存
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 2. 拷贝到显存：目标在前，源在后
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 3. 启动 kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);

    // 4. 拷贝回内存
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 5. 释放显存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
```

`h_` 开头的指针指向内存，`d_` 开头的指向显存。host 代码不能通过 `*d_a` 或 `d_a[i]` 直接读取显存中的数据，指向显存的指针只能交给 CUDA API，或作为参数传给 kernel。

## 错误检查

CUDA API 返回 `cudaError_t` 类型的错误码，成功时为 `cudaSuccess`。kernel 启动没有返回值，要在启动后立即调用 `cudaGetLastError()` 查询。实际项目中常把检查封装成宏：

```cpp
#define CUDA_CHECK(call)                                  \
    do {                                                  \
        cudaError_t err = (call);                         \
        if (err != cudaSuccess) {                         \
            printf("%s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err));              \
            exit(1);                                      \
        }                                                 \
    } while (0)

CUDA_CHECK(cudaMalloc(&d_a, size));
vecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
CUDA_CHECK(cudaGetLastError());
```

## 三个函数限定符

| 限定符       | 执行位置 | 调用方                          | 说明                                      |
| ------------ | -------- | ------------------------------- | ----------------------------------------- |
| `__global__` | device   | host                            | kernel；每次调用都会启动一个新的 grid     |
| `__device__` | device   | kernel 或其他 `__device__` 函数 | 普通函数调用，不创建新线程；host 无法调用 |
| `__host__`   | host     | host                            | 普通 CPU 函数；不加限定符时默认为此       |

`__host__` 与 `__device__` 可以同时使用，编译器会生成两个版本，分别供 CPU 和 GPU 调用。

## 本课的编程题

- `solve` 接收的指针已经指向显存：分配、拷贝到显存、拷贝回内存和释放都由判题程序完成，你不用写 host 函数的第 1、2、4、5 步。
- 有的题给定了 `solve`，只需编写 kernel；有的题要自己写启动配置。以题面的「实现要求」为准。
- 判题用例会特意选取不能被常用线程数整除的 `n`，漏写边界检查必然出错。

:::tip
整数相除会舍去小数部分，在 kernel 中同样如此。`9 / 5` 的两个操作数都是整数，结果为 `1` 而非 `1.8`。设 `x` 为 `float`，`x * 9 / 5` 没有问题：先算乘法，结果为 `float`，随后的除法也是浮点除法；而 `9 / 5 * x` 会先得到整数 `1`。需要小数结果时，应写成 `9.0f / 5.0f`。
:::

## 一个线程计算多个元素

视频中每个线程计算一个元素，也可以让一个线程计算多个元素。此时需要重新推导下标公式，思路与全局编号相同，依次回答两个问题：

1. 之前的各 block 共占用多少个位置？（每个 block 的线程数不变，但每个线程负责的元素变多了。）
2. 本线程负责的几个元素，在所属 block 占用的区段中从哪里开始？

求出第一个元素的下标后，每个元素都要单独与 `n` 比较。例如每个线程计算相邻的两个元素、`n` 为奇数时，负责数组末尾的线程只有一个有效元素，它的第二个元素必须由边界检查排除。

分配方式也不止一种：一个线程可以负责相邻的两个元素，也可以负责数组前半段和后半段中各一个元素。就结果而言，两种方式都正确；二者在速度上的差异，留到「kernel 性能调优」一章讨论合并访存时再作比较。
