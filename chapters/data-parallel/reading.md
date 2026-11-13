## What Work Can a GPU Take On

A GPU does not try to finish one task especially fast. It tries to finish a huge number of tasks at the same time. So what kind of computation can be handed to a huge number of threads at once?

Think of grading exams: a hundred exams can be graded by a hundred assistants at the same time, because each exam is graded on its own and nobody needs to look at another exam. Many computations in programs work the same way. The simplest example is **vector addition**: take two arrays `a` and `b`, both of length `n`, add the two numbers at the same position, and store the sum at the same position of an array `c`.

```
c[i] = a[i] + b[i]
```

Computing `c[i]` needs only `a[i]` and `b[i]`; no other position is involved. So the `n` elements can be computed in any order, or all at the same time.

When the computation on each piece of data is independent of the others and can run at the same time, we call it **data parallelism**. It is everywhere:

- Changing the brightness of a photo: millions of pixels, each computed on its own.
- Simulating many particles: at every step, each particle's new position is computed on its own.
- Adding a million numbers into one sum: it looks like the numbers must be added one after another, but you can split them into many short pieces, sum each piece separately, and then combine the results (the _Reducing an Array to One Value_ chapter covers this).

The other kind is **task parallelism**: several different jobs run at the same time, like a media player that decodes video while it plays audio. How many parallel parts you get depends on how many independent jobs the program is split into; more data does not give you more. Data parallelism is different: the more data, the more pieces can run at once, without writing a single extra line of code. Double the width and the height of a photo, and there are four times as many pixels, so four times as many threads can work at once. GPU performance comes mainly from data parallelism.

Writing a data-parallel program means organizing the computation around the data, so that independent pieces of work go to different threads that run at the same time. In real programs the data parallelism is often well hidden and you have to find it first. Most of this course is practice in exactly that skill.

## From a for Loop to a Kernel

Start with the most familiar version: vector addition in plain C, on the CPU.

```cpp
void vecAdd(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

This is the sequential version: with a million elements the loop iterates a million times, and only one element is being computed at any moment. A stream of instructions executed one after another like this is a **thread**.

### Host and device

A CUDA program has two sides: the CPU side is called the **host**, and the GPU side is called the **device**. Code for both can live in the same source file (with the `.cu` extension). Most of the code is still ordinary C++ running on the host; only the part handed to the GPU is marked with CUDA-specific keywords. So an ordinary C++ program can be seen as a CUDA program with host code only. CUDA adds just a few pieces of new syntax and a set of library functions on top of C++.

### Kernels

The part handed to the GPU is a function called a **kernel**. A kernel is executed on the GPU by a large number of threads together. Turning the sequential function above into a kernel takes two changes:

```cpp
__global__ void vecAddKernel(const float* a, const float* b, float* c, int n) {
    int i = /* which element does this thread compute? filled in below */;
    c[i] = a[i] + b[i];
}
```

1. The keyword `__global__` (two underscores on each side) goes in front of the function. It means: this is a kernel, called from the host and executed on the device. A kernel must return `void`.
2. The `for` loop is gone. The body keeps only the line that computes one element.

The `n` iterations have not disappeared. When the kernel is launched, CUDA starts a large number of threads at once, and each thread runs the function body once, taking the place of one iteration of the old loop. Handing each iteration of a loop to its own thread is called **loop parallelism**. All threads run the same code but work on different data. This style is called **SPMD** (Single Program, Multiple Data). Each thread on its own still executes its statements in order, top to bottom.

::motion{path="learn-cuda/loop-to-threads"}

Isn't one thread per element too expensive? Starting a thread on a CPU is indeed not cheap. On a GPU, threads are created and scheduled by hardware, which makes them very cheap; launching millions of threads at once is routine.

## Which Element Does Each Thread Compute

All threads run the same code, yet each thread needs a different `i`. Otherwise every thread would compute the same element and the rest would never be computed. How does a thread know which element is its own?

### Grids, blocks and threads

CUDA does not start threads one by one. It starts them in groups:

- Each group is a **block**. Within one launch, every block has the same number of threads. A block holds at most 1024 threads; 256 is a common choice, and a multiple of 32 is best (the reason has to do with how the hardware executes threads, covered in the _How a GPU Runs Your Threads_ chapter).
- All the blocks of one launch together form a **grid**.

Every thread can read a few variables that CUDA sets up in advance:

| Variable      | Meaning                                                                         |
| ------------- | ------------------------------------------------------------------------------- |
| `threadIdx.x` | Thread index: the thread's number inside its own block, starting at 0           |
| `blockIdx.x`  | Block index: the number of the block the thread is in; same for all its threads |
| `blockDim.x`  | The number of threads in each block                                             |

These variables are read-only. The `.x` suffix is there because threads can also be arranged in 2D or 3D (next chapter). The data in this chapter is one-dimensional, so `x` is all we need.

### The global index

Work through a small example: 4 threads per block, and look at thread 2 of block 1. Block 0 in front of it has 4 threads, which already take positions 0 to 3, so its `i` is `1 × 4 + 2 = 6`.

| block | thread index | `i`       |
| ----- | ------------ | --------- |
| 0     | 0 1 2 3      | 0 1 2 3   |
| 1     | 0 1 2 3      | 4 5 6 7   |
| 2     | 0 1 2 3      | 8 9 10 11 |

::motion{path="learn-cuda/global-index"}

In general:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

`blockIdx.x * blockDim.x` is the number of positions taken by all the blocks before this one; adding `threadIdx.x` gives the thread's position in the whole grid. Block 0 covers 0 to 3, block 1 covers 4 to 7, block 2 continues after that, end to end, with nothing skipped.

`i` is a local variable in the kernel. Every thread has its own copy, invisible to the others. The same line of code reads different indices in different threads, so each thread gets a different `i` and computes a different element of the array. That is what makes SPMD work: one program, with the indices splitting up the data.

## Extra Threads: the Bounds Check

`n` is not always an exact multiple of the threads per block. Take `n = 10` with 4 threads per block: to give all 10 elements a thread, you need 3 blocks, 12 threads in total. The last two threads get `i` equal to 10 and 11, but the array has no such positions. If they write anyway, they write past the end of the array, corrupting other data or even crashing the program.

::motion{path="learn-cuda/out-of-bounds"}

So the kernel needs an `if`: work only when `i` is less than `n`. Extra threads do nothing and simply finish. The complete kernel:

```cpp
__global__ void vecAddKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

:::warning
Forgetting `if (i < n)` is the most common beginner mistake. It goes unnoticed when `n` happens to be a multiple of the threads per block, and writes out of bounds for any other `n`. The test cases in this course deliberately use values of `n` that common block sizes do not divide evenly.
:::

## Launching a Kernel: Configuration and Rounding Up

Calling a kernel looks like calling an ordinary function, except that the function name is followed by an **execution configuration** wrapped in `<<<` and `>>>`. The first value is the number of blocks; the second is the number of threads per block.

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;
vecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
```

The `d_a`, `d_b` and `d_c` passed in parentheses are pointers to device memory; how the data gets there is explained in _Moving Data into Device Memory_ below.

The threads per block is usually a fixed value, such as 256; the number of blocks has to follow `n`. Writing `n / threads` directly is a bug: in C, dividing two integers throws away the fraction. With 10 elements and 4 threads per block, `10 / 4` is 2, so only 2 blocks and 8 threads are launched, the last two elements get no thread, and the result is wrong. You need to **round up**. The usual way is to add `threads - 1` to `n` before dividing:

```
(10 + 4 - 1) / 4 = 13 / 4 = 3
```

Exactly 3 blocks. With this formula, the threads per block can stay fixed while the number of blocks follows `n` automatically. A million elements with 256 threads per block gives 3907 blocks; the 192 extra threads are stopped by the `if` in the kernel.

:::tip
Integer division dropping the fraction can bite inside a kernel too. In `9 / 5` both sides are integers, so the result is `1`, not `1.8`. `x * 9 / 5` is fine: `x` is a `float`, the multiplication comes first and gives a `float`, so the division is a floating-point division. But `9 / 5 * x` computes the integer `1` first. If you want the fraction, write `9.0f / 5.0f`.
:::

## The Order of Blocks

CUDA makes no promise about which block runs first. Blocks may run one after another in any order, or several at the same time. Your code must not assume any order between blocks.

This rule buys a big advantage. Because blocks are independent of each other's order, the hardware can run as many of them at once as it likes. A smaller GPU runs fewer at a time and works through them batch by batch; a larger GPU runs many more at once and spreads them out in one go. The same program, without changing a line, runs faster on a bigger GPU.

## Several Elements per Thread

The version above computes one element per thread, but that is not the only way to split the work: one thread can compute several elements. The index formula then has to be worked out again, using the same reasoning as above. Ask the same two questions again:

1. How many positions do the blocks before this one take in total? (The threads per block stay the same, but each thread now covers more elements.)
2. Where, inside the stretch its block covers, do this thread's elements start?

Once you have the index of the first element, every further element needs its own check against `n`. For example, with two elements per thread and an odd `n`, the last thread has only one element to compute, and its second element must be skipped.

There is also more than one way to pick the elements: a thread can take two neighbouring elements, or one from the first half of the array and one from the second half. Both give correct results; how they differ in speed is compared in the _Tuning Kernel Performance_ chapter, where memory coalescing is covered.

## Moving Data into Device Memory

### Host memory and device memory

The data a program works on, whether read from a file or received over the network, starts out in host memory on the CPU side. The GPU has its own memory, **device memory**. The two are separate and connected by a bus, commonly PCIe. In CUDA, device memory is formally called **global memory**; why "global" is explained in the _Shared Memory and Tiling_ chapter. Any data a kernel reads or writes has to be in device memory first.

### The five host steps

To have the GPU do one vector addition, the host goes through five steps:

1. Allocate three regions of device memory for `a`, `b` and `c`.
2. Copy `a` and `b` from host memory to device memory.
3. Launch the kernel to do the computation.
4. Copy the result `c` from device memory back to host memory.
5. Free the device memory.

::motion{path="learn-cuda/host-steps"}

Step 3 uses the execution configuration from _Launching a Kernel_. Each of the other four steps is a function provided by CUDA (an API).

After launching a kernel, the host does not stop to wait for it; it moves on right away. The step that actually waits is step 4: the copy back does not start until the kernel has finished. So the `c` that comes back is always the finished result.

::motion{path="learn-cuda/async-launch"}

### Moving data is expensive

How long does all this copying take? We measured it on an H100 with `n = 2²⁵` (about 33.5 million elements, 128 MB per array), taking the median of 7 runs.

| What                               | Time    |
| ---------------------------------- | ------- |
| CPU, one core, the sequential loop | 37 ms   |
| GPU, all five steps                | 39 ms   |
| of which: copy to device           | 29.4 ms |
| of which: copy back to host        | 8.1 ms  |
| of which: the kernel itself        | 0.19 ms |

The whole GPU run takes about as long as the CPU, and almost all of the time goes into moving data; the kernel itself is nearly two hundred times faster than the CPU loop. That is why real programs do not copy back and forth for every computation. They keep large data in device memory, run several kernels in a row, and copy back only at the end.

### Allocating and freeing: `cudaMalloc`, `cudaFree`

```cpp
float* d_a;
cudaMalloc(&d_a, n * sizeof(float));
// ...
cudaFree(d_a);
```

`cudaMalloc` is deliberately similar to C's `malloc`. It takes two arguments:

- The second is the number of bytes. `n` floats at 4 bytes each is `n * sizeof(float)`.
- The first is not `d_a` but **the address of `d_a`**, `&d_a`. C's `malloc` hands you the address as its return value, but `cudaMalloc` keeps its return value for reporting errors (see _Checking for Errors_). So it works the other way: you give it the address of the variable `d_a`, and it writes the address of the allocated device memory straight into `d_a`. Older code often writes `cudaMalloc((void**)&d_a, size)`; it does the same thing, and the cast is no longer needed.

::motion{path="learn-cuda/device-pointer"}

`cudaFree` does not need to write anything into `d_a`; it only needs the address, so you pass `d_a` itself.

A naming habit: pointers to host memory start with `h_`, pointers to device memory start with `d_`, so you can tell them apart at a glance.

:::warning
`d_a` holds a device memory address. Host code must **not** read the data it points to with `*d_a` or `d_a[i]`: that memory is not in host memory, and reading it makes the program fail. Pointers to device memory can only be passed to CUDA API functions or to kernels.
:::

### Copying: `cudaMemcpy`

Steps 2 and 4 use the same function, `cudaMemcpy`. Its four arguments are the destination, the source, the number of bytes and the direction, with **the destination first and the source second**. The direction is a constant:

| Direction     | Constant                 |
| ------------- | ------------------------ |
| host → device | `cudaMemcpyHostToDevice` |
| device → host | `cudaMemcpyDeviceToHost` |

```cpp
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);  // to the device
cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);  // back to the host
```

### The complete host function

Putting the five steps together:

```cpp
void vecAdd(const float* h_a, const float* h_b, float* h_c, int n) {
    int size = n * sizeof(float);
    float *d_a, *d_b, *d_c;

    // 1. Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 2. Copy to the device
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

This function has the same name and parameter types as the sequential version we started with. Whoever calls it does not need to know that it uses a GPU.

:::note
In this course's coding problems, the pointers passed to `solve` already point to device memory: the judge does the allocating, copying and freeing for you. You only write step 3, the kernel and its launch.
:::

## Checking for Errors

The code above leaves out error checking to stay readable. In real programs you cannot skip it.

Every CUDA API function returns an error code of type `cudaError_t`. Success is `cudaSuccess`; anything else should be reported. For example, `cudaMalloc` fails when there is not enough device memory:

```cpp
cudaError_t err = cudaMalloc(&d_a, size);
if (err != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(err));
}
```

A kernel launch is different: it has no return value. Ask for 2000 threads per block, more than the limit of 1024, and the launch fails, but it does not report anything by itself. The program carries on and ends up with wrong results. So right after a launch, call `cudaGetLastError()` to ask whether anything went wrong:

```cpp
vecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(err));
}
```

Real projects usually wrap these checks in a macro and apply it to every call.

`cudaGetLastError()` only catches errors in the launch itself, such as the invalid configuration above. Errors that happen while the kernel runs (for example, an out-of-bounds access to an invalid address) are different. The host does not wait for the kernel to finish (see _The five host steps_), so when `cudaGetLastError()` is called the kernel may still be running and such an error is not yet known. It is reported the next time the host waits for the GPU, for example in the return value of the `cudaMemcpy` in step 4. Check that return value too; if no `cudaMemcpy` follows, call `cudaDeviceSynchronize()` to wait for the kernel explicitly and check its return value.

## Compiling: nvcc

A `.cu` file mixes two kinds of code, one for the CPU and one for the GPU, plus keywords like `__global__` and syntax like `<<<>>>` that an ordinary C++ compiler does not understand. So it is compiled with CUDA's own compiler, **nvcc**:

1. nvcc first splits the file into two halves by its keywords.
2. The host half goes to the system's ordinary C++ compiler and becomes CPU machine code.
3. nvcc compiles the device half itself into GPU machine code, and also keeps a copy as **PTX** (an intermediate code for GPUs).
4. Both halves go into the final program.

At run time, the driver picks the machine code that matches the GPU. If the program runs on a newer GPU with no matching machine code, the driver compiles the PTX on the spot.

## The Three Function Qualifiers

| Qualifier    | Runs on | Called from                               | Notes                                                         |
| ------------ | ------- | ----------------------------------------- | ------------------------------------------------------------- |
| `__global__` | device  | host                                      | A kernel; every call starts a new grid                        |
| `__device__` | device  | a kernel or another `__device__` function | An ordinary call, no new threads; the host cannot call it     |
| `__host__`   | host    | host                                      | An ordinary CPU function; the default when nothing is written |

`__host__` and `__device__` can be used together: the compiler then builds two versions, callable from both the CPU and the GPU.

## Summary

The CUDA extensions used in this chapter fall into just four groups:

1. Function qualifiers: `__global__`, `__device__`, `__host__`.
2. The kernel launch syntax `<<<blocks, threads>>>`.
3. Variables set up in advance: `threadIdx`, `blockIdx`, `blockDim`.
4. APIs for device memory and data movement: `cudaMalloc`, `cudaMemcpy`, `cudaFree`, plus `cudaGetLastError` and `cudaDeviceSynchronize` for error checking.

This chapter covers only the small part you need now; other features come later when they are needed. What really matters is the parallel thinking behind it: **writing a kernel means writing only the work of one thread, which is one iteration of the old loop.** The whole loop becomes a large number of threads all running the same kernel; each thread uses its own index to find its own element, and an `if` stops the extra threads.
