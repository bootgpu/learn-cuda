<!-- prettier-ignore -->
::youtube[Chapter 3 video: Covering 2D and 3D Data with Threads]{id="_rCf-vHDLUU"}

This page collects the code and formulas from the video, so you can look them up while solving the problems.

## Thread Indices in 2D and 3D

```cpp
int col   = blockIdx.x * blockDim.x + threadIdx.x;  // column: x direction
int row   = blockIdx.y * blockDim.y + threadIdx.y;  // row: y direction
int plane = blockIdx.z * blockDim.z + threadIdx.z;  // plane: z direction
```

In every direction the index is "block index × block size + thread index", the same as the global index in the previous chapter. Every direction can run past the edge, so every direction needs a check:

```cpp
if (col < width && row < height) { /* 2D */ }
if (col < width && row < height && plane < depth) { /* 3D */ }
```

## Launch Configuration: Round Up in Each Direction

2D:

```cpp
dim3 block(16, 16);  // a missing value is 1, i.e. (16, 16, 1)
dim3 grid((width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y);
kernel<<<grid, block>>>(/* arguments */);
```

3D:

```cpp
dim3 block(8, 8, 4);
dim3 grid((width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y,
          (depth + block.z - 1) / block.z);
kernel<<<grid, block>>>(/* arguments */);
```

:::warning
The first value of a `dim3` is x, the horizontal direction: it goes with **columns** and **width**. This is the opposite of the rows-first habit of "3 rows by 4 columns" or `a[row][col]`, and it is easy to get backwards.
:::

Size limits:

- A block holds at most 1024 threads in total (the product of its three sizes); each direction has its own limit as well: x and y at most 1024, z at most 64.
- A grid can have at most 2³¹ − 1 blocks in x, and at most 65535 each in y and z.

## Flattened Indices

| Layout                                | Index                                        |
| ------------------------------------- | -------------------------------------------- |
| 2D, row-major                         | `row * width + col`                          |
| 2D, column-major                      | `col * height + row`                         |
| 3D, plane by plane, row-major in each | `plane * width * height + row * width + col` |

C and C++ 2D arrays are row-major; NVIDIA's matrix library cuBLAS is column-major.

## Color to Grayscale

Each pixel stores R, G and B in three neighbouring bytes; a grayscale pixel is one byte:

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
        unsigned char r = in[i * CHANNELS];      // pixel i starts at byte i × 3
        unsigned char g = in[i * CHANNELS + 1];
        unsigned char b = in[i * CHANNELS + 2];
        out[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

// host side
dim3 block(16, 16);
dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
rgbToGrayKernel<<<grid, block>>>(d_in, d_out, width, height);
```

## Box Blur

Each output pixel is the average of a patch of input pixels centred on it. Where the patch runs past the edge of the image, only the pixels inside the image are added, and the sum is divided by how many were actually added.

```cpp
#define BLUR_SIZE 1  // patch radius: the patch is 2 * BLUR_SIZE + 1 wide

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
                    ++count;  // count only pixels inside the image
                }
            }
        }
        out[row * width + col] = sum / count;
    }
}
```

The host-side launch is the same as for the grayscale kernel. The input and the output must be two separate buffers in device memory.

## Matrix Multiplication

P = M × N, where all three are n × n square matrices stored row-major. The element of P in row `row`, column `col` is the dot product of row `row` of M and column `col` of N:

```
P[row][col] = M[row][0] × N[0][col] + M[row][1] × N[1][col] + … + M[row][n−1] × N[n−1][col]
```

On the CPU, with three nested loops:

```cpp
for (int row = 0; row < n; ++row)
    for (int col = 0; col < n; ++col) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
            sum += M[row * n + k] * N[k * n + col];
        P[row * n + col] = sum;
    }
```

On the GPU, each thread computes one element of P:

```cpp
__global__ void matMulKernel(const float* M, const float* N, float* P, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += M[row * n + k] * N[k * n + col];  // row `row` of M, column `col` of N
        }
        P[row * n + col] = sum;
    }
}

// host side
dim3 block(16, 16);
dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
matMulKernel<<<grid, block>>>(d_M, d_N, d_P, n);
```

The two outer loops (`row`, `col`) have independent iterations, so they became threads; the inner loop (`k`) adds to the same `sum` on every iteration, so it stays inside the thread.
