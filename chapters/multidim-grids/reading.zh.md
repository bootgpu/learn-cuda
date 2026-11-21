::bilibili[第 3 章视频：用线程铺满二维、三维数据]{bvid="BV1EisH6sEG3"}

本页汇总视频中的代码与公式，做题时可以直接对照。

## 二维、三维的线程编号

```cpp
int col   = blockIdx.x * blockDim.x + threadIdx.x;  // 列号：x 方向
int row   = blockIdx.y * blockDim.y + threadIdx.y;  // 行号：y 方向
int plane = blockIdx.z * blockDim.z + threadIdx.z;  // 层号：z 方向
```

每个方向的编号都等于「block 编号 × block 尺寸 + 线程编号」，与上一章的全局编号相同。每个方向都可能越界，都要检查：

```cpp
if (col < width && row < height) { /* 二维 */ }
if (col < width && row < height && plane < depth) { /* 三维 */ }
```

## 启动配置：每个方向分别向上取整

二维：

```cpp
dim3 block(16, 16);  // 省略的方向为 1，即 (16, 16, 1)
dim3 grid((width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y);
kernel<<<grid, block>>>(/* 参数 */);
```

三维：

```cpp
dim3 block(8, 8, 4);
dim3 grid((width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y,
          (depth + block.z - 1) / block.z);
kernel<<<grid, block>>>(/* 参数 */);
```

:::warning
`dim3` 的第一个数是 x，即横向，对应**列**和**宽**，与「3 行 4 列」、`a[row][col]` 这类先行后列的习惯相反，很容易写反。
:::

尺寸上限：

- 一个 block 的线程数，即三个方向之积，不超过 1024；各方向另有上限：x、y 不超过 1024，z 不超过 64。
- grid：x 方向最多 2³¹ − 1 个 block，y、z 方向各最多 65535 个。

## 摊平后的下标

| 存放方式               | 下标                                         |
| ---------------------- | -------------------------------------------- |
| 二维，行优先           | `row * width + col`                          |
| 二维，列优先           | `col * height + row`                         |
| 三维，逐层、层内行优先 | `plane * width * height + row * width + col` |

C 和 C++ 的二维数组采用行优先；NVIDIA 的矩阵库 cuBLAS 采用列优先。

## 彩色图转灰度

每个像素的 R、G、B 各占 1 个字节，依次相邻存放；灰度图每个像素 1 个字节：

```
gray = 0.299 × r + 0.587 × g + 0.114 × b
```

```cpp
#define CHANNELS 3

__global__ void rgbToGrayKernel(const unsigned char* in, unsigned char* out,
                                int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int i = row * width + col;
        unsigned char r = in[i * CHANNELS];      // 第 i 个像素从第 i × 3 个字节开始
        unsigned char g = in[i * CHANNELS + 1];
        unsigned char b = in[i * CHANNELS + 2];
        out[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

// host 端
dim3 block(16, 16);
dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
rgbToGrayKernel<<<grid, block>>>(d_in, d_out, width, height);
```

## 均值模糊

每个输出像素取输入图中以它为中心的一块像素的平均值。块越过图片边缘时，只累加图内的像素，并除以实际累加的个数。

```cpp
#define BLUR_SIZE 1  // 块的半径：块的边长为 2 * BLUR_SIZE + 1

__global__ void blurKernel(const unsigned char* in, unsigned char* out,
                           int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int sum = 0;
        int count = 0;
        for (int nRow = row - BLUR_SIZE; nRow <= row + BLUR_SIZE; ++nRow) {
            for (int nCol = col - BLUR_SIZE; nCol <= col + BLUR_SIZE; ++nCol) {
                if (nRow >= 0 && nRow < height && nCol >= 0 && nCol < width) {
                    sum += in[nRow * width + nCol];
                    ++count;  // 只数图内的像素
                }
            }
        }
        out[row * width + col] = sum / count;
    }
}
```

host 端的启动配置与灰度 kernel 相同。输入和输出必须是两块不同的显存。

## 矩阵乘法

P = M × N，三个都是 n × n 的方阵，按行优先存放。P 第 `row` 行、第 `col` 列的元素，是 M 第 `row` 行与 N 第 `col` 列的点积：

```
P[row][col] = M[row][0] × N[0][col] + M[row][1] × N[1][col] + … + M[row][n−1] × N[n−1][col]
```

CPU 上用三重循环计算：

```cpp
for (int row = 0; row < n; ++row)
    for (int col = 0; col < n; ++col) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
            sum += M[row * n + k] * N[k * n + col];
        P[row * n + col] = sum;
    }
```

GPU 上每个线程计算 P 的一个元素：

```cpp
__global__ void matMulKernel(const float* M, const float* N, float* P, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += M[row * n + k] * N[k * n + col];  // M 的第 row 行、N 的第 col 列
        }
        P[row * n + col] = sum;
    }
}

// host 端
dim3 block(16, 16);
dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
matMulKernel<<<grid, block>>>(d_M, d_N, d_P, n);
```

外层两个循环（`row`、`col`）各轮互不依赖，换成了线程；最内层循环（`k`）每轮都累加到同一个 `sum` 上，留在线程内部。
