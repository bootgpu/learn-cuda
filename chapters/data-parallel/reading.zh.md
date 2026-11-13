::bilibili[第 2 章视频：从循环到 kernel]{bvid="BV1nkW962EJS"}

## 哪些计算适合交给 GPU

GPU 的设计目标不是缩短单项任务的耗时，而是同时完成大量任务。那么，什么样的计算能够同时交给大量线程执行？

以批改试卷为例：一百份试卷可以由一百名助教同时批改，因为各份试卷的批改相互独立，批改其中一份无须参考其他试卷。程序中的许多计算具有同样的性质，最简单的例子是**向量加法**：给定两个长度均为 `n` 的数组 `a` 和 `b`，将同一位置上的两个元素相加，结果写入数组 `c` 的对应位置。

```
c[i] = a[i] + b[i]
```

计算 `c[i]` 只用到 `a[i]` 和 `b[i]`，与其他位置无关。因此，这 `n` 个元素可以按任意顺序计算，也可以全部同时计算。

像这样，对各份数据的计算互不依赖、可以同时进行，称为**数据并行**（data parallelism）。数据并行的计算十分常见：

- 调整照片亮度：照片有数百万个像素，各像素的计算相互独立。
- 模拟大量粒子的运动：在每个时间步中，各粒子新位置的计算相互独立。
- 对一百万个数求和：看似只能逐个累加，实际上也可以拆成许多小段，先分别求出各段之和，再将结果合并（「把数组归约成一个值」一章专门讨论）。

另一种并行称为**任务并行**（task parallelism）：多项不同的任务同时进行，例如播放器同时解码画面和播放声音。任务并行能同时进行的份数，取决于程序能拆分出多少项互不依赖的任务，不随数据量增长。数据并行则不同：数据量越大，能同时进行的计算就越多，而且无须增加代码。照片的长和宽各增加一倍，像素数变为四倍，能同时工作的线程数也随之变为四倍。GPU 的性能主要来自数据并行。

编写数据并行程序，就是围绕数据重新组织计算，将互不依赖的计算分配给不同线程同时执行。实际程序中的数据并行往往不那么明显，需要先识别出来；本课程的大部分练习都围绕这一能力展开。

## 从 for 循环到 kernel

先看最熟悉的写法：用普通的 C 语言在 CPU 上实现向量加法。

```cpp
void vecAdd(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

这是串行写法：一百万个元素需要循环迭代一百万次，同一时刻只计算一个元素。像这样按顺序逐条执行的指令流，称为一个**线程**（thread）。

### host 和 device

CUDA 程序分为两部分：CPU 一侧称为 **host**，GPU 一侧称为 **device**。两部分代码可以写在同一个源文件中（扩展名为 `.cu`）。大部分代码仍是普通的 C++，在 host 上运行；只有交给 GPU 执行的部分需要用 CUDA 专用的关键字标明。因此，普通的 C++ 程序可以看作只含 host 代码的 CUDA 程序。CUDA 只在 C++ 的基础上增加了少量新语法和库函数。

::motion{path="learn-cuda/host-device"}

### kernel

交给 GPU 执行的部分是一个函数，称为 **kernel**。kernel 在 GPU 上由大量线程共同执行。将上面的串行函数改写为 kernel，需要两处改动：

```cpp
__global__ void vecAddKernel(const float* a, const float* b, float* c, int n) {
    int i = /* 本线程负责第几个元素？见下一节 */;
    c[i] = a[i] + b[i];
}
```

1. 函数前加关键字 `__global__`（前后各两个下划线），表示这是一个 kernel：由 host 调用，在 device 上执行。kernel 的返回类型必须是 `void`。
2. 去掉 `for` 循环，循环体 `c[i] = a[i] + b[i];` 原样保留。原来由循环变量提供的 `i`，改由每个线程根据自己的编号算出（见下一节）。

原来的 `n` 次迭代并没有消失。启动 kernel 时，CUDA 一次创建大量线程，每个线程执行一遍函数体，恰好对应原来的一次迭代。这种把循环的每次迭代分配给一个线程的做法，称为**循环并行**（loop parallelism）。所有线程执行同一份代码，但各自处理不同的数据，这种方式称为 **SPMD**（Single Program, Multiple Data，单程序多数据）。就单个线程而言，代码仍是自上而下逐条顺序执行的。

::motion{path="learn-cuda/loop-to-threads"}

为每个元素创建一个线程，开销是否过大？在 CPU 上创建线程的代价确实较高；而 GPU 上线程的创建和调度由硬件完成，代价很低，一次启动数百万个线程十分常见。

## 每个线程计算哪个元素

所有线程执行同一份代码，但每个线程的 `i` 必须各不相同；否则所有线程都会计算同一个元素，其余元素无人计算。那么，每个线程如何确定自己负责第几个元素？

### grid、block 和线程

CUDA 的线程不是逐个零散创建的，而是分组组织：

- 每组称为一个 **block**。同一次启动中，所有 block 的线程数相同。一个 block 最多包含 1024 个线程，常用 256，最好取 32 的倍数（原因与硬件的执行方式有关，见「GPU 如何执行线程」一章）。
- 一次启动的全部 block 合称一个 **grid**。

每个线程都可以读取以下几个由 CUDA 预先设置的内置变量：

| 变量          | 含义                                                         |
| ------------- | ------------------------------------------------------------ |
| `threadIdx.x` | 线程编号：本线程在所属 block 中的编号，从 0 开始             |
| `blockIdx.x`  | block 编号：本线程所属 block 的编号，同一 block 内各线程相同 |
| `blockDim.x`  | 每个 block 的线程数                                          |

这些变量是只读的。之所以带后缀 `.x`，是因为线程还可以按二维、三维排列（下一章介绍）；本章的数据是一维的，只用到 `x`。

::motion{path="learn-cuda/grid-blocks"}

### 全局编号

先看一个小例子：每个 block 有 4 个线程，考察 1 号 block 中的 2 号线程。它之前的 0 号 block 有 4 个线程，已占用 0 到 3 这四个位置，因此它的 `i` 为 `1 × 4 + 2 = 6`。

| block | 线程编号 | `i`       |
| ----- | -------- | --------- |
| 0     | 0 1 2 3  | 0 1 2 3   |
| 1     | 0 1 2 3  | 4 5 6 7   |
| 2     | 0 1 2 3  | 8 9 10 11 |

::motion{path="learn-cuda/global-index"}

写成一般形式：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

`blockIdx.x * blockDim.x` 是之前各 block 共占用的位置数，再加上 `threadIdx.x`，即为本线程在整个 grid 中的位置。0 号 block 负责 0 到 3，1 号 block 负责 4 到 7，2 号 block 依次向后，各段首尾相接，不重不漏。

`i` 是 kernel 中的局部变量，每个线程各有一份，彼此不可见。执行同一行代码时，不同线程读到的编号不同，算出的 `i` 也不同，从而各自处理数组中的不同元素。SPMD 正是依靠这一点实现的：同一份程序，通过编号分配到不同的数据。

## 边界检查

`n` 不一定恰好是每个 block 线程数的整数倍。例如 `n = 10`、每个 block 4 个线程：要让 10 个元素都有线程计算，需要 3 个 block，共 12 个线程。最后两个线程算出的 `i` 为 10 和 11，而数组中并没有这两个位置。如果让它们照常写入，就会越过数组边界，破坏其他数据，甚至导致程序崩溃。

::motion{path="learn-cuda/out-of-bounds"}

因此 kernel 中需要加一条 `if` 判断：只有 `i` 小于 `n` 时才执行计算，多出的线程不做任何操作，直接结束。完整的 kernel 如下：

```cpp
__global__ void vecAddKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

:::warning
漏写 `if (i < n)` 是初学者最常见的错误。当 `n` 恰好是每个 block 线程数的整数倍时，问题不会暴露；换一个 `n` 就会越界。本课程的判题用例会特意选取无法被常用线程数整除的 `n`。
:::

## 启动 kernel：启动配置与向上取整

调用 kernel 的写法与调用普通函数类似，只是函数名后多了一段由 `<<<` 和 `>>>` 括起的**启动配置**（execution configuration）：第一个参数是 block 的个数，第二个参数是每个 block 的线程数。

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;
vecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
```

圆括号中传入的 `d_a`、`d_b`、`d_c` 是指向 GPU 显存的指针；数据如何放入显存，见后文「在内存与显存之间传输数据」一节。

每个 block 的线程数通常取固定值，如 256；block 的个数则需根据 `n` 计算。直接写 `n / threads` 是错误的：C 语言中两个整数相除，小数部分会被直接舍去。仍以 10 个元素、每个 block 4 个线程为例，`10 / 4` 的结果为 2，只启动 2 个 block、8 个线程，最后两个元素无人计算，结果出错。因此需要**向上取整**，常见写法是先将 `n` 加上 `threads - 1`，再做除法：

```
(10 + 4 - 1) / 4 = 13 / 4 = 3
```

恰好得到 3 个 block。借助这个式子，每个 block 的线程数可以固定不变，block 的个数自动随 `n` 调整。例如一百万个元素、每个 block 256 个线程，需要启动 3907 个 block；多出的 192 个线程由 kernel 中的 `if` 判断排除。

:::tip
整数除法舍去小数的问题在 kernel 中同样会遇到。`9 / 5` 的两个操作数都是整数，结果为 `1` 而非 `1.8`。设 `x` 为 `float`，`x * 9 / 5` 没有问题：先算乘法，结果为 `float`，随后的除法也是浮点除法；而 `9 / 5 * x` 会先得到整数 `1`。需要小数结果时，应写成 `9.0f / 5.0f`。
:::

## block 的执行顺序

CUDA 不保证各 block 的执行顺序：它们可能以任意次序依次执行，也可能多个同时执行。因此，代码不能对 block 的执行顺序做任何假设。

这一约定带来一个重要的好处：正因为 block 之间没有顺序要求，硬件可以自行决定同时执行多少个 block。规模较小的 GPU 同时执行的 block 较少，其余 block 分批依次执行；规模较大的 GPU 能同时执行更多 block。同一份程序无须修改，在更大的 GPU 上即可自动获得更高的速度。

## 一个线程计算多个元素

上面的写法让每个线程计算一个元素，但这并非唯一的分配方式，也可以让一个线程计算多个元素。此时需要重新推导下标公式，思路与前面相同，依次回答两个问题：

1. 之前的各 block 共占用多少个位置？（每个 block 的线程数不变，但每个线程负责的元素变多了。）
2. 本线程负责的几个元素，在所属 block 占用的区段中从哪里开始？

求出第一个元素的下标后，每个元素都要单独与 `n` 比较。例如每个线程计算相邻的两个元素、`n` 为奇数时，负责数组末尾的线程只有一个有效元素，它的第二个元素必须由边界检查排除。

分配方式也不止一种：一个线程可以负责相邻的两个元素，也可以负责数组前半段和后半段中各一个元素。就结果而言，两种方式都正确；二者在速度上的差异，留到「kernel 性能调优」一章讨论合并访存时再作比较。

## 在内存与显存之间传输数据

### 内存和显存

程序要处理的数据，无论读自文件还是来自网络，最初都位于 CPU 一侧的内存中。GPU 有自己独立的存储器，称为**显存**。内存与显存相互独立，二者通过总线相连，常见的是 PCIe。显存在 CUDA 中的正式名称是 **global memory**（全局内存），「全局」一词的含义留到「共享内存与分块」一章说明。kernel 运行时读写的数据，都必须先放入显存。

### host 端的五个步骤

要让 GPU 完成一次向量加法，host 端需要执行五个步骤：

1. 在显存中为 `a`、`b`、`c` 分配三块空间。
2. 将 `a` 和 `b` 从内存拷贝到显存。
3. 启动 kernel，由线程执行计算。
4. 将计算结果 `c` 从显存拷贝回内存。
5. 释放显存。

::motion{path="learn-cuda/host-steps"}

第 3 步使用「启动 kernel」一节介绍的启动配置，其余四步各对应一个 CUDA 提供的 API 函数。

kernel 启动后，host 不会等待其执行完毕，而是立即继续执行后续代码。真正发生等待的是第 4 步：拷贝回内存的操作要等 kernel 执行完毕后才开始。因此，拷贝回来的 `c` 一定是计算完成的结果。

::motion{path="learn-cuda/async-launch"}

### 数据传输的开销

数据在内存与显存之间来回传输，耗时有多少？我们在一块 H100 上做了实测：`n = 2²⁵`（约 3355 万个元素，每个数组 128 MB），重复 7 次取中位数。

| 做法                 | 耗时    |
| -------------------- | ------- |
| CPU 单核运行串行循环 | 37 ms   |
| GPU 完整执行五个步骤 | 39 ms   |
| 其中：拷贝到显存     | 29.4 ms |
| 其中：拷贝回内存     | 8.1 ms  |
| 其中：kernel 执行    | 0.19 ms |

GPU 全流程的耗时与 CPU 相当，时间几乎全部用于数据传输；kernel 本身则比 CPU 循环快近两百倍。因此，实际程序不会每计算一次就来回拷贝数据，而是让大块数据常驻显存，连续执行多个 kernel，最后才将结果拷贝回内存。

### 分配和释放：`cudaMalloc`、`cudaFree`

```cpp
float* d_a;
cudaMalloc(&d_a, n * sizeof(float));
// ...
cudaFree(d_a);
```

`cudaMalloc` 的设计有意与 C 语言的 `malloc` 保持相似，它有两个参数：

- 第二个参数是要分配的字节数。`n` 个 `float`、每个 4 字节，即 `n * sizeof(float)`。
- 第一个参数传入的不是 `d_a`，而是 **`d_a` 的地址** `&d_a`。C 的 `malloc` 通过返回值交出分配到的地址，而 `cudaMalloc` 的返回值用于报告是否出错（见「错误检查」一节）。因此它采用另一种方式：调用者传入变量 `d_a` 的地址，`cudaMalloc` 将分配到的显存地址直接写入 `d_a`。旧代码常写成 `cudaMalloc((void**)&d_a, size)`，作用相同，如今已无须这一强制类型转换。

::motion{path="learn-cuda/device-pointer"}

`cudaFree` 无须修改 `d_a`，只需知道要释放的地址，因此直接传入 `d_a`。

命名惯例：指向内存的指针以 `h_`（host）开头，指向显存的指针以 `d_`（device）开头，便于区分。

:::warning
`d_a` 中存放的是显存地址。在 host 代码中**不能**通过 `*d_a` 或 `d_a[i]` 读取它指向的数据：该地址不在内存中，读取会导致程序出错。指向显存的指针只能交给 CUDA API，或作为参数传给 kernel。
:::

### 拷贝：`cudaMemcpy`

第 2 步和第 4 步使用同一个函数 `cudaMemcpy`。它的四个参数依次为：目标地址、源地址、字节数、拷贝方向，注意**目标在前，源在后**。拷贝方向用以下常量表示：

| 方向        | 常量                     |
| ----------- | ------------------------ |
| 内存 → 显存 | `cudaMemcpyHostToDevice` |
| 显存 → 内存 | `cudaMemcpyDeviceToHost` |

```cpp
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);  // 拷贝到显存
cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);  // 拷贝回内存
```

### 完整的 host 函数

将五个步骤组合起来：

```cpp
void vecAdd(const float* h_a, const float* h_b, float* h_c, int n) {
    int size = n * sizeof(float);
    float *d_a, *d_b, *d_c;

    // 1. 分配显存
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 2. 拷贝到显存
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

这个函数的名称和参数类型与最初的串行版本完全相同，调用者无须知道其内部使用了 GPU。

:::note
本课程的编程题中，`solve` 接收的指针已经指向显存：分配、拷贝到显存、拷贝回内存和释放均由判题程序完成，你只需编写第 3 步，即 kernel 及其启动代码。
:::

## 错误检查

为突出主干，上面的代码省略了错误检查。实际编程时，错误检查不可省略。

CUDA API 函数一般返回一个类型为 `cudaError_t` 的错误码：成功时为 `cudaSuccess`，否则应输出错误原因。例如显存不足时，`cudaMalloc` 会失败：

```cpp
cudaError_t err = cudaMalloc(&d_a, size);
if (err != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(err));
}
```

kernel 启动则不同，它没有返回值。例如每个 block 请求 2000 个线程，超过 1024 的上限，启动就会失败；但这一失败不会主动报告，程序照常执行，最终得到错误的结果。因此，启动后应立即调用 `cudaGetLastError()`，查询启动是否出错：

```cpp
vecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(err));
}
```

实际项目中，通常将这些检查封装成宏，应用于每一次调用。

`cudaGetLastError()` 只能发现启动阶段的错误，例如上面这种不合法的启动配置。kernel 执行过程中发生的错误（例如越界访问了非法地址）则不同：kernel 启动后 host 不等待其执行完毕（见「host 端的五个步骤」），调用 `cudaGetLastError()` 时 kernel 可能尚未执行完，这类错误还无从得知。它们要等 host 下一次等待 GPU 时才会报告，例如第 4 步 `cudaMemcpy` 的返回值。因此，第 4 步的返回值同样需要检查；如果之后没有 `cudaMemcpy`，可以调用 `cudaDeviceSynchronize()` 显式等待 kernel 执行完毕，并检查它的返回值。

## 编译：nvcc

一个 `.cu` 文件中同时包含在 CPU 上运行的代码和在 GPU 上运行的代码，还有 `__global__` 这样的关键字和 `<<<>>>` 这样的新语法，普通的 C++ 编译器无法识别。因此需要使用 CUDA 自带的编译器 **nvcc**：

1. nvcc 首先根据关键字将源文件拆分为 host 代码和 device 代码两部分。
2. host 代码交给系统中普通的 C++ 编译器，编译为 CPU 机器码。
3. device 代码由 nvcc 自行编译为 GPU 机器码，同时保留一份 **PTX**（一种 GPU 中间代码）。
4. 两部分的编译产物一起打包进最终的可执行程序。

运行时，驱动程序选用与当前 GPU 匹配的机器码。如果换用更新的 GPU，程序中没有与之匹配的机器码，驱动程序会在运行时将 PTX 即时编译为该 GPU 的机器码。

## 三个函数限定符

| 限定符       | 执行位置 | 调用方                          | 说明                                      |
| ------------ | -------- | ------------------------------- | ----------------------------------------- |
| `__global__` | device   | host                            | kernel；每次调用都会启动一个新的 grid     |
| `__device__` | device   | kernel 或其他 `__device__` 函数 | 普通函数调用，不创建新线程；host 无法调用 |
| `__host__`   | host     | host                            | 普通 CPU 函数；不加限定符时默认为此       |

`__host__` 与 `__device__` 可以同时使用，编译器会生成两个版本，分别供 CPU 和 GPU 调用。

## 小结

本章用到的 CUDA 扩展共四类：

1. 函数限定符：`__global__`、`__device__`、`__host__`。
2. 启动 kernel 的语法 `<<<blocks, threads>>>`。
3. 内置变量：`threadIdx`、`blockIdx`、`blockDim`。
4. 管理显存与传输数据的 API：`cudaMalloc`、`cudaMemcpy`、`cudaFree`，以及用于错误检查的 `cudaGetLastError`、`cudaDeviceSynchronize`。

本章只介绍了 CUDA 中够用的一小部分，其余功能留待后续章节按需介绍。真正的重点在于背后的并行思路：**编写 kernel，就是编写单个线程的工作，即原来循环中的一次迭代。** 整个循环由大量线程取代，所有线程执行同一个 kernel，每个线程依据自己的编号找到负责的元素，并通过 `if` 判断排除多出的线程。
