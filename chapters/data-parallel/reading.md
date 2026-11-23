::bilibili[Chapter 2 video: Your First CUDA Program]{bvid="BV1nkW962EJS"}

The video is in Chinese for now; an English version is coming.

This page collects the code and formulas from the video, so you can look them up while solving the problems. The last two sections cover what the problems need but the video does not go into.

## The Kernel: Global Index and Bounds Check

```cpp
__global__ void vecAddKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // global index
    if (i < n) {                                    // extra threads do nothing
        c[i] = a[i] + b[i];
    }
}
```

| Variable      | Meaning                                             |
| ------------- | --------------------------------------------------- |
| `threadIdx.x` | this thread's index within its block, starting at 0 |
| `blockIdx.x`  | the index of this thread's block, starting at 0     |
| `blockDim.x`  | the number of threads per block                     |

`blockIdx.x * blockDim.x` is the number of positions taken by all the blocks before this one; adding `threadIdx.x` gives the thread's position in the whole grid. `n` is not always a multiple of the threads per block, and `if (i < n)` stops the extra threads.

## Launch Configuration: Rounding Up

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;  // round up
vecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
```

The first value in `<<<…>>>` is the number of blocks; the second is the number of threads per block. A block holds at most 1024 threads; 256 is common. Writing `n / threads` throws away the fraction, and the last few elements get no thread.

## The Complete Host Function

```cpp
void vecAdd(const float* h_a, const float* h_b, float* h_c, int n) {
    int size = n * sizeof(float);
    float *d_a, *d_b, *d_c;

    // 1. Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 2. Copy to the device: destination first, source second
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 3. Launch the kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);

    // 4. Copy back to the host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 5. Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
```

Pointers starting with `h_` point to host memory; pointers starting with `d_` point to device memory. Host code cannot read device data directly with `*d_a` or `d_a[i]`; pointers to device memory can only be passed to CUDA API functions or to kernels.

## Checking for Errors

CUDA API functions return an error code of type `cudaError_t`, which is `cudaSuccess` on success. A kernel launch has no return value, so call `cudaGetLastError()` right after the launch. Real projects usually wrap the checks in a macro:

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

## The Three Function Qualifiers

| Qualifier    | Runs on | Called from                               | Notes                                                         |
| ------------ | ------- | ----------------------------------------- | ------------------------------------------------------------- |
| `__global__` | device  | host                                      | A kernel; every call starts a new grid                        |
| `__device__` | device  | a kernel or another `__device__` function | An ordinary call, no new threads; the host cannot call it     |
| `__host__`   | host    | host                                      | An ordinary CPU function; the default when nothing is written |

`__host__` and `__device__` can be used together: the compiler then builds two versions, callable from both the CPU and the GPU.

## The Coding Problems in This Course

- The pointers passed to `solve` already point to device memory: the judge does the allocating, copying and freeing, so you do not write steps 1, 2, 4 and 5 of the host function.
- Some problems give you `solve` and you only write the kernel; others ask you to write the launch configuration yourself. Follow the problem's _Implementation Requirements_.
- The test cases deliberately use values of `n` that common block sizes do not divide evenly, so a missing bounds check always fails.

:::tip
Integer division drops the fraction inside a kernel too. In `9 / 5` both sides are integers, so the result is `1`, not `1.8`. `x * 9 / 5` is fine: `x` is a `float`, the multiplication comes first and gives a `float`, so the division is a floating-point division. But `9 / 5 * x` computes the integer `1` first. If you want the fraction, write `9.0f / 5.0f`.
:::

## Several Elements per Thread

In the video each thread computes one element, but one thread can also compute several. The index formula then has to be worked out again, using the same reasoning as the global index. Answer two questions in turn:

1. How many positions do the blocks before this one take in total? (The threads per block stay the same, but each thread now covers more elements.)
2. Where, inside the stretch its block covers, do this thread's elements start?

Once you have the index of the first element, every element needs its own check against `n`. For example, with two neighbouring elements per thread and an odd `n`, the last thread has only one element to compute, and its second element must be stopped by the bounds check.

There is also more than one way to pick the elements: a thread can take two neighbouring elements, or one from the first half of the array and one from the second half. Both give correct results; how they differ in speed is compared in the _Tuning Kernel Performance_ chapter, where memory coalescing is covered.
