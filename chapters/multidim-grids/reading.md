The previous chapter worked on one-dimensional arrays: threads were laid out in one direction, each thread computed an index `i` with `blockIdx.x * blockDim.x + threadIdx.x`, and handled element `i`. Much data, however, is not one-dimensional. An image is made of rows and columns of pixels, so it is two-dimensional; a video is a sequence of images, one frame after another, so it is three-dimensional. This chapter covers how threads should be arranged when the data is 2D or 3D, and how each thread finds the data it is responsible for. Along the way we build three examples: converting a color image to grayscale, blurring an image, and multiplying matrices.

## Arranging Threads in 2D and 3D

CUDA threads can already be arranged in three dimensions: the threads of a block can be laid out along x, y and z, and so can the blocks of a grid. x is horizontal, y is vertical, and z can be thought of as layers stacked one behind another.

Take the grid in the figure below. Each block has 4 threads across, 4 down and only 1 layer; in x, y, z order this is written (4, 4, 1). The whole grid has 3 blocks across, 2 down and again only 1 layer, written (3, 2, 1). With a single layer in z, both the blocks and the grid here are two-dimensional.

::figure[A grid of 3 × 2 blocks, each block 4 × 4 threads]{id="learn-cuda/grid-2d"}

These two triples are exactly what goes into the launch configuration: the first is the grid, the second is the block. Each triple has the type `dim3` and is usually defined as a variable before the launch:

```cpp
dim3 grid(3, 2, 1);
dim3 block(4, 4, 1);
kernel<<<grid, block>>>(/* arguments */);
```

A few notes:

- A direction with a single layer is written as 1, never 0. The number of threads is the product of the three values; a 0 would mean no threads at all, which is an invalid launch configuration, and the launch fails.
- Directions left out default to 1: `dim3 block(4, 4)` is the same as `dim3 block(4, 4, 1)`. The single integer in the previous chapter's launch configurations works the same way: `256`, for example, becomes `(256, 1, 1)`, a one-dimensional block that uses only x.
- The block size is given only once, so every block in a grid has the same size. All 6 blocks in the figure are 4 × 4.

### Built-in variables in three directions

Since threads are laid out in three directions, the built-in variables have three components too, `.x`, `.y` and `.z`:

| Variable    | Meaning                                | The circled thread in the figure |
| ----------- | -------------------------------------- | -------------------------------- |
| `threadIdx` | this thread's index within its block   | (1, 2, 0)                        |
| `blockIdx`  | the index of its block within the grid | (2, 1, 0)                        |
| `blockDim`  | the size of a block                    | (4, 4, 1)                        |

All indices start at 0. Inside its block, the circled thread has 1 thread to its left and 2 above it, so `threadIdx.x` is 1 and `threadIdx.y` is 2. Its block has 2 blocks to its left and 1 above, so `blockIdx.x` is 2 and `blockIdx.y` is 1. There is only one layer, so `.z` is 0 for both.

How many dimensions to use usually follows the shape of the data: for an array, both the grid and the block are one-dimensional; for an image, both are laid out in 2D, which makes the index arithmetic most direct. The grid and the block do not have to have the same number of dimensions.

## Which Pixel Does Each Thread Handle

Take a small image 10 pixels wide and 6 pixels high, again with one thread per pixel. With 4 × 4 blocks, start tiling from the top-left corner of the image. Horizontally, one block covers 4 columns and two cover 8, leaving 2 columns, so 3 blocks are needed across. Vertically, one block covers 4 rows, leaving 2 rows, so 2 blocks are needed down. That is exactly the (3, 2, 1) grid from the previous section.

Next, work out which pixel each thread is responsible for, that is, the pixel's **column** (col) and **row** (row), both counted from 0. Look at the circled thread in the figure below: its block has `blockIdx` (1, 1), and within that block its `threadIdx` is (2, 1).

- Column: counting from the left, there is one whole block before it, taking up 4 columns; inside its own block it passes 2 more threads. The column is `1 × 4 + 2 = 6`.
- Row: counting from the top, there is one whole block before it, taking up 4 rows; inside its own block it passes 1 more. The row is `1 × 4 + 1 = 5`.

So this thread handles the pixel in row 5, column 6.

::figure[The circled thread handles the pixel in row 5, column 6]{id="learn-cuda/thread-to-pixel"}

This is exactly how `i` was computed in the previous chapter: the positions taken up by the blocks before, plus this thread's position within its own block. The x direction gives the column, and the y direction gives the row:

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

Each thread plugs in its own indices and gets the row and column it is responsible for.

### Bounds checks in two directions

The grid has 3 blocks across, 12 columns of threads in total, and 2 blocks down, 8 rows of threads. The image, however, has only 10 columns and 6 rows. There are 2 extra columns of threads on the right and 2 extra rows at the bottom, and their positions fall outside the image. The thread in the top-right corner, for example, has column 11, while the largest column in the image is 9.

::figure[12 × 8 threads cover a 10 × 6 image; the extra threads fall outside the image]{id="learn-cuda/grid-overhang"}

The previous chapter used `if (i < n)` to stop the extra threads. In 2D either direction can go out of range, so both conditions are checked together:

```cpp
if (col < width && row < height) {
    // handle the pixel in row `row`, column `col`
}
```

This example launches 12 × 8 = 96 threads, and exactly 10 × 6 = 60 of them pass the check, one per pixel; the other 36 do nothing.

## Launch Configuration: Round Up in Each Direction

On the host, the launch is configured with two `dim3` variables. The block size is a fixed value; the number of blocks in each direction of the grid uses the round-up from the previous chapter, based on the width horizontally and the height vertically:

```cpp
dim3 block(16, 16);
dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
kernel<<<grid, block>>>(/* arguments */);
```

Check it with the small image from before. With 4 × 4 blocks, horizontally `(10 + 4 - 1) / 4 = 13 / 4 = 3` and vertically `(6 + 4 - 1) / 4 = 9 / 4 = 2`, which is exactly (3, 2). Real programs usually use 16 × 16 blocks, which is 256 threads. The expression uses `block.x` and `block.y` rather than literal numbers, so nothing else needs to change when the block size changes.

The names `grid` and `block` can be anything you like; `threadIdx`, `blockIdx` and `blockDim` in the kernel are names defined by CUDA and cannot be changed.

:::warning
The first value of a `dim3` is x, the horizontal direction, which corresponds to **columns** and the image's **width**. This is the opposite of row-first habits such as "3 rows by 4 columns" or `a[row][col]`, and it is easy to swap them. For the same 10-wide, 6-high image, the grid should be (3, 2); written row-first as (2, 3), there are only 2 blocks and 8 columns of threads across, so columns 8 and 9 are never handled and the result is wrong. The extra threads in the vertical direction are stopped by the `if`, so the program reports no error, and this kind of mistake is hard to notice.
:::

### Size limits for blocks and grids

- The number of threads in a block, the product of the three directions, cannot exceed 1024. 16 × 16 = 256 is fine; 32 × 32 = 1024 is exactly at the limit.
- Each direction also has its own limit: x and y at most 1024, z at most 64.
- The grid limits are much looser: up to 2³¹ − 1 blocks in x, and up to 65535 in each of y and z. You will rarely reach them.

A launch configuration that exceeds a limit makes the launch fail. As described in the previous chapter's _Checking for Errors_ section, such a failure is not reported on its own; you have to call `cudaGetLastError()` after the launch to find out.

## Flattening 2D Data into 1D

With its row and column, a thread now needs to read the corresponding pixel. The kernel parameter `in` points to the image in device memory, and the most natural thing to write is to treat it as a 2D array: `in[row][col]`. That does not compile.

The reason is that memory is one-dimensional: every byte has an address, and the addresses start at 0 and increase in a single direction. The elements of a 2D array can only be stored one after another in memory as well. When you write `a[row][col]`, the compiler has to turn it into "the how-many-th element from the start", and to do that it has to skip the whole rows before, so it must know how many elements a row has, and know it at compile time: when a 2D array is a parameter, its number of columns must be a constant, as in `float a[][10]`. But the same kernel has to handle images of every size, and the width is passed in as a parameter only at run time. The compiler cannot know the row length, so the program has to do this conversion itself.

:::details[Why not use an array of pointers]
There is another way in C to write two pairs of brackets: allocate an array of pointers, each pointing to one row; `a[row]` fetches the pointer to row `row`, and `[col]` then locates the element within that row. On a GPU this is expensive: every row needs its own `cudaMalloc` call, the pointer array itself must be copied to device memory too, and every element read needs an extra read of the row pointer first.
:::

So the usual approach is to allocate a single contiguous region of device memory, place the elements of the 2D data into it one after another, and let the program compute the index itself. This is called **flattening**, also known as **linearization**.

### Row-major

The most common way to flatten is to store one row after another: row 0 first, then row 1 right after it, and so on. This is called **row-major** order. Take a matrix with 3 rows and 4 columns. The element in row 1, column 2 has one whole row before it, taking up 4 positions, and passes 2 more within its own row, so its flattened index is `1 × 4 + 2 = 6`. The reasoning is the same as for the column: the positions taken up by the whole chunks before, plus the position within the chunk, except that here the "chunk" is a whole row.

The number of elements in a row is the matrix's width, so the row-major index is `row * width + col`. The `in[row][col]` that did not compile becomes:

```cpp
in[row * width + col]
```

### Column-major

You can also store one column after another, which is called **column-major** order. Look again at the element in row 1, column 2: there are two whole columns before it, 3 elements each, and it passes 1 more within its own column, so its index is `2 × 3 + 1 = 7`. The number of elements in a column is the matrix's height, so the column-major index is `col * height + row`.

::figure[A 3 × 4 matrix flattened in row-major and column-major order; the element in row 1, column 2 lands at index 6 and 7 respectively]{id="learn-cuda/flatten-2d"}

2D arrays in C and C++ are row-major. NVIDIA's matrix library cuBLAS is column-major, for compatibility with Fortran programs. When the two layouts are mixed without matching them up, you get the wrong element: the same index 7 is row 1, column 2 in column-major order, but row 1, column 3 in row-major order.

### From an index back to row and column

Sometimes all you have is a flattened index and you need the row and column back, for example when threads are numbered in 1D but the data is 2D. In row-major order, the elements before index `i` fill some number of whole rows, each with `width` elements:

```cpp
int row = i / width;   // how many whole rows come before it
int col = i % width;   // how many more within its own row
```

For the same 3 × 4 matrix, index 6 gives `6 / 4 = 1` and `6 % 4 = 2`: row 1, column 2, matching the earlier calculation. In column-major order each column has `height` elements, and rows and columns swap roles: `col = i / height`, `row = i % height`. Index 7 gives `7 / 3 = 2` and `7 % 3 = 1`: column 2, row 1.

### 3D data

3D data is flattened the same way. Threads gain an index in the z direction, called the **plane**, computed just like the row and column, but with `.z`:

```cpp
int plane = blockIdx.z * blockDim.z + threadIdx.z;
```

When flattening, all of plane 0 comes first, then plane 1, and each plane is stored row-major inside. Take data with 2 planes of 3 rows and 4 columns each: a plane has `3 × 4 = 12` elements, width × height. The element in plane 1, row 1, column 2 has one whole plane before it, taking up 12 positions, plus its index 6 within its own plane, so its flattened index is `12 + 6 = 18`. In general, one more term is added in front of the row-major expression:

```cpp
plane * width * height + row * width + col
```

::figure[Data with 2 planes of 3 rows and 4 columns flattened plane by plane; the element in plane 1, row 1, column 2 lands at index 18]{id="learn-cuda/flatten-3d"}

All three directions can go out of range, so the bounds check has three conditions, and the host rounds up separately in all three directions. Let the number of planes (the depth) be `depth`:

```cpp
// in the kernel
if (col < width && row < height && plane < depth) {
    // handle the element in plane `plane`, row `row`, column `col`
}

// on the host
dim3 block(8, 8, 4);
dim3 grid((width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y,
          (depth + block.z - 1) / block.z);
```

Here the block's z size is 4, within the limit of 64, and the product of the three is 8 × 8 × 4 = 256, within 1024.

## Converting a Color Image to Grayscale

The previous sections covered three things: how to lay threads out in 2D, how each thread computes its own row and column, and how to turn a row and column into a flattened index. Putting the three together gives the first kernel that works on 2D data: converting a color image to a grayscale image.

### How pixels are stored

Each pixel of a color image has three **channels**: red (R), green (G) and blue (B). Each channel is an integer from 0 to 255 and takes 1 byte (`unsigned char`), so a pixel takes 3 bytes. Once the whole image is flattened row-major, memory holds the R, G, B of the first pixel, followed by the R, G, B of the next pixel, and so on, interleaved.

Each pixel of a grayscale image has only 1 byte, its brightness, computed as:

```
gray = 0.299 × r + 0.587 × g + 0.114 × b
```

Green has the largest coefficient because the human eye is most sensitive to green. For an orange pixel with color (251, 148, 10), for example, the result is about 163.07; storing it in an `unsigned char` drops the fractional part, giving a gray value of 163. Each pixel is computed independently of the others, which fits one thread per pixel exactly.

### The kernel

```cpp
#define CHANNELS 3

__global__ void rgbToGrayKernel(const unsigned char* in, unsigned char* out,
                                int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int i = row * width + col;
        unsigned char r = in[i * CHANNELS];
        unsigned char g = in[i * CHANNELS + 1];
        unsigned char b = in[i * CHANNELS + 2];
        out[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}
```

The parameter `in` is the color image, `out` is the grayscale image, and `width` and `height` are the image's width and height. The first two lines compute the column and row, the `if` stops the threads outside the image, and then the flattened index `i` is computed. Each grayscale pixel takes 1 byte, so `i` can be used directly. Each color pixel takes 3 bytes: the `i` pixels before pixel `i` take `i × 3` bytes, so it starts at byte `i × 3`, and the 3 bytes from there are its R, G and B in order.

With the 10-wide, 6-high image from before, the thread handling row 5, column 6 computes `i = 5 × 10 + 6 = 56`; in the color image this pixel starts at byte `56 × 3 = 168`, and bytes 168, 169 and 170 are its R, G and B.

### The host side

The launch configuration on the host is the same as before:

```cpp
dim3 block(16, 16);
dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
rgbToGrayKernel<<<grid, block>>>(d_in, d_out, width, height);
```

`d_in` and `d_out` are pointers to device memory. Real image sizes are usually not multiples of 16, so the rightmost column of blocks and the bottom row of blocks each have some threads outside the image, and the `if` stops them.

## Blurring an Image

In the grayscale conversion, each thread reads just the 3 bytes of one pixel and does a few multiply-adds. The next example blurs an image, and each thread has to read a whole patch of surrounding pixels.

Blurring smooths out sharp changes between neighbouring pixels: fine detail fades, while large outlines remain. Photos often contain scattered bright or dark specks called **noise**. After blurring, a speck differs much less from the pixels around it and no longer stands out, which is why blurring is often used for denoising.

### Mean blur

The simplest blur takes an average: for each pixel of the output image, take the patch of the input image centred on the same position, for example 3 × 3, which is 9 pixels, and compute their average.

Below are the pixel values of the top-left 5 rows and 5 columns of a grayscale image. The 250 in row 2, column 2 is a bright speck of noise; the pixels around it are all about 50:

```
 52  47  50  46  53
 55  52  55  50  49
 49  48 250  53  51
 51  51  49  54  50
 48  53  50  47  52
```

The 3 × 3 patch centred on the 250 pixel is the 9 pixels in rows 1 to 3 and columns 1 to 3. Their sum is 662, and 662 divided by 9 is about 73.56; dropping the fractional part gives 73. The speck drops from 250 to 73. Each of the 8 pixels around it also has the 250 in its own 3 × 3 patch, and all of them come out as 72 or 73. What was a single isolated bright dot becomes a 3 × 3 area only slightly brighter than its surroundings, and the sharp change is smoothed out.

:::details[Weighted average]
In a mean blur, all 9 pixels in the patch count equally. Blurs used in practice usually give each pixel a **weight**, how many shares it gets in the average, with larger weights closer to the centre. For example, the centre gets 4 shares, the pixels above, below, left and right get 2 each, and the four corners get 1 each: multiply each pixel by its weight, add them up, and divide by the sum of the weights, 16. This is a weighted average, and the widely used Gaussian blur is of this kind. A computation like this, summing a patch of data according to weights, is called a **convolution**, and it is the subject of the _Convolution Filters_ chapter.
:::

### The blur kernel

Again, each thread computes one output pixel:

```cpp
#define BLUR_SIZE 1  // patch radius: the patch is 2 * BLUR_SIZE + 1 on each side

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
                    ++count;
                }
            }
        }
        out[row * width + col] = sum / count;
    }
}
```

The parameter `in` is a grayscale image with 1 byte per pixel, so the index does not need to be multiplied by 3 as it was for the color image. The first three lines, computing the column and row and the `if`, are exactly the same as in the grayscale kernel.

The patch size is set by the constant `BLUR_SIZE`, the patch's **radius**: how many cells it extends outward from the centre. With radius 1, it extends 1 cell to the left and 1 to the right, plus the centre cell, so each side is 2 × 1 + 1 = 3 and the patch is 3 × 3; with radius 2, the patch is 5 × 5.

What is new is the pair of loops in the middle, which read the pixels of the patch one by one. `nRow` and `nCol` are the row and column of the position being read, to keep them apart from the thread's own `row` and `col`. The outer loop takes `nRow` from `row - BLUR_SIZE` to `row + BLUR_SIZE`, and the inner loop takes `nCol` from `col - BLUR_SIZE` to `col + BLUR_SIZE`. For the thread handling the 250 pixel, `row` and `col` are both 2, and `nRow` and `nCol` each take the values 1, 2, 3, giving 9 positions. For each position, the row-major index `nRow * width + nCol` is computed, the pixel value is read and added to `sum`, and `count` goes up by 1. When the loops finish, `sum` is 662 and `count` is 9; `sum / count` is integer division, which drops the fractional part and gives 73, written to the same position of the output image. What the `if` inside the loops and `count` are for is explained in the next subsection.

Two more points:

- The input and output must be two separate regions of device memory. Every thread needs the original values of its neighbouring pixels; if results were written straight back into the input image, a neighbouring thread could read a value that has already been overwritten.
- The previous chapter replaced the loop in the sequential code with a batch of threads, yet loops appear in this kernel again. Which loops should become threads and which should stay inside a thread is discussed when we get to matrix multiplication.

The launch configuration on the host is the same as for the grayscale kernel.

### When the patch goes past the image edge

When the centre pixel is on the edge of the image, the patch reaches outside the image. Take the top-left pixel: its row and column are both 0, and its 3 × 3 patch covers rows −1 to 1 and columns −1 to 1. Five of those positions have row or column −1 and are not in the image.

::figure[5 positions of the top-left pixel's 3 × 3 patch fall outside the image]{id="learn-cuda/blur-edge"}

Without a check, reading directly at `nRow * width + nCol` gets the wrong data at these positions:

- Row 1, column −1: the index is `1 × width − 1`, the element just before the start of row 1. The image is stored row-major, so this is the last pixel of row 0, on the other side of the image and not a neighbour of the current pixel at all.
- Row −1: the index is `−1 × width + nCol`. Since `nCol` is less than `width`, the result is negative, before the start of the array: an out-of-bounds access.

So before reading each position, check that it lies inside the image; that is the `if` inside the loops:

```cpp
if (nRow >= 0 && nRow < height && nCol >= 0 && nCol < width)
```

Compared with the check at the start of the kernel, this one has two extra "not less than 0" conditions. The thread's own `row` and `col` are computed from non-negative indices and cannot be below 0, whereas `nRow` and `nCol` are `row` and `col` minus the radius and can be negative.

Once positions outside the image are skipped, different pixels add up different numbers of values: a corner pixel has only 4 positions inside the image, an edge pixel that is not a corner has 6, and only interior pixels have the full 9. Dividing by 9 every time would give the top-left pixel (52 + 47 + 55 + 52) / 9 = 206 / 9, which is 22, far below its original value of 52. Edge pixels likewise come out too dark because the divisor is too large, and the whole image gets a dark border.

The right approach is to divide by the number of values actually added. `++count` sits inside the `if`, so positions outside the image are not counted and `count` is exactly the number of pixels added. The top-left pixel gets 206 / 4, which is 51, close to its original 52. The centre pixel itself is always inside the image, so `count` is at least 1 and there is never a division by 0.

The grayscale kernel checks only once, at the start; the blur kernel does an extra check for every pixel it reads. This leads to a general habit: every time you read or write an array, make sure the index is within the valid range.

## Matrix Multiplication

In the blur, each thread reads 9 or more pixels. The last example in this chapter is matrix multiplication: each thread multiplies a whole row of one matrix by a whole column of another.

Matrix multiplication is everywhere in linear algebra, and many algorithms for solving equations and other numerical computations are built on it; most of the computation in training and running neural networks is matrix multiplication too. Different implementations of the same operation can differ in speed by several orders of magnitude. This section writes the simplest one; faster versions are left to the _Faster Matrix Multiplication_ chapter.

:::details[The three levels of BLAS]
The most common linear algebra operations have a standard library interface called BLAS (Basic Linear Algebra Subprograms). cuBLAS, mentioned in the _Column-major_ section, is NVIDIA's implementation of it on the GPU. BLAS has three levels, depending on whether the operands are vectors or matrices: level 1 is vector-vector operations, such as the previous chapter's vector addition; level 2 is matrix-vector multiplication; level 3 is matrix-matrix multiplication, the example in this section.
:::

### What matrix multiplication computes

Let P = M × N. The element of P in row `row`, column `col` is computed from row `row` of M and column `col` of N: starting from the beginning of both, multiply the first element by the first element, the second by the second, and so on to the last, then add all these products together. This operation is called a **dot product**.

A dot product pairs up the elements of both sides one to one, so a row of M must have as many elements as a column of N; that is, the number of columns of M must equal the number of rows of N. The matrices do not have to be square. For example, if M has 4 rows and 6 columns and N has 6 rows and 3 columns, a row of M and a column of N both have 6 elements, so they can be multiplied. Each row of M corresponds to a row of P and each column of N to a column of P, so P has 4 rows and 3 columns: as many rows as M and as many columns as N.

To keep things simple, this section deals only with square matrices: M, N and P are all n rows by n columns. Written out, the dot product is:

```
P[row][col] = M[row][0] × N[0][col] + M[row][1] × N[1][col] + … + M[row][n−1] × N[n−1][col]
```

::figure[With n = 6, the element of P in row 1, column 4 is the dot product of row 1 of M and column 4 of N]{id="learn-cuda/matmul-dot"}

### One thread per element of P

P has n² elements, each needs one dot product, and the dot products do not depend on each other. Following the same approach as before, each thread computes one element of P. Each thread computes its row and column the same way as in the grayscale kernel; the resulting (`row`, `col`) both identifies the thread and gives the position in P of the element it is responsible for.

With n = 6, for example, take 3 × 3 blocks and a 2 × 2 grid, which covers the 6 × 6 P exactly. Each block is responsible for one 3 × 3 region of P; such a region is called a **tile**, and P is divided into 4 tiles. This concept comes back in the _Shared Memory and Tiling_ chapter.

```cpp
__global__ void matMulKernel(const float* M, const float* N, float* P, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += M[row * n + k] * N[k * n + col];
        }
        P[row * n + col] = sum;
    }
}
```

The first three lines, computing the column and row and the bounds check, are the same as in the grayscale kernel, except that the width and height are both n. Next comes the dot product: `sum` starts at 0, `k` goes from 0 to n − 1, and each step multiplies element `k` of row `row` of M by element `k` of column `col` of N and adds the product to `sum`. When the loop finishes, `sum` is written to row `row`, column `col` of P.

All three matrices are flattened row-major with width n, so every index is "row × n + column":

- The element read from M is in row `row`, column `k`, at index `row * n + k`. Each time `k` goes up by 1, the index goes up by 1, so the thread reads n consecutive elements in memory. For the thread handling row 1, column 4 with n = 6, the indices are 6, 7, 8, 9, 10, 11.
- The element read from N is in row `k`, column `col`, at index `k * n + col`. Each time `k` goes up by 1, the index goes up by n, skipping a whole row each time. The same thread reads N at indices 4, 10, 16, 22, 28, 34.
- P is written at `row * n + col`, the same expression as `row * width + col` in the grayscale kernel.

::figure[The thread handling row 1, column 4 of P reads indices 6 to 11 of the flattened M one after another, and skips 6 elements at a time in the flattened N]{id="learn-cuda/matmul-index"}

The launch configuration on the host is the same as for the grayscale kernel, with both width and height replaced by n:

```cpp
dim3 block(16, 16);
dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
matMulKernel<<<grid, block>>>(d_M, d_N, d_P, n);
```

### Which loops become threads

The blur section left a question open: which loops should become threads, and which should stay inside a thread. On a CPU, matrix multiplication is written as three nested loops, which make a good place to answer it:

```cpp
for (int row = 0; row < n; ++row)
    for (int col = 0; col < n; ++col) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
            sum += M[row * n + k] * N[k * n + col];
        P[row * n + col] = sum;
    }
```

The outer two loops run over rows and columns; each iteration computes a different element of P, and the iterations do not depend on each other, so both became threads. In the previous chapter, a whole loop was replaced by a batch of threads; the loops that can be replaced that way are exactly these, whose iterations are independent. The innermost loop is different: every iteration adds a product to the same `sum` and has to wait for the previous addition to finish, so it stays inside the thread and runs in order.

The innermost loop can be split up too, but only with several threads working together to accumulate the same sum. This kind of computation is called a **reduction**, and it is the subject of the _Reducing an Array to One Value_ chapter. The two loops in the blur kernel also accumulate into the same `sum`, so they stay inside the thread. The grayscale conversion does not give the three channels to three separate threads for the same reason: the three channels are combined into a single gray value, so one thread computes it.

## Another Split: One Thread per Row

One thread per element of P is not the only way to split the work. As the previous chapter's _Several Elements per Thread_ section showed, one thread can also be responsible for several elements. For matrix multiplication, a natural choice is to have one thread compute a whole row of P.

The threads then only need to be laid out in 1D, n of them: thread `row` is responsible for row `row` of P, and `row` is just the global index from the previous chapter:

```cpp
int row = blockIdx.x * blockDim.x + threadIdx.x;
```

Inside the thread, another loop walks through every column of that row; each column is an independent dot product, the `k` loop of the kernel above. The threads are laid out in a single direction, so the bounds check has only one condition left: `row < n`.

The split changes only which elements each thread is responsible for; how each element is computed stays the same. The element of P in row `row`, column `col` is still the dot product of row `row` of M and column `col` of N, and the indices are still `row * n + k`, `k * n + col` and `row * n + col`.

The same idea applies to any computation where each row produces one result. Matrix-vector multiplication is an example: element `row` of the result is the dot product of row `row` of the matrix with the vector, so one thread handles one row and loops through it. When the matrix is not square, note that the number of elements in a row is the matrix's width (its number of columns), whereas the number of threads and the bounds check use the matrix's height (its number of rows).

The price of this split is fewer threads: the thread count drops from n² to n, and each thread does n times as much work. A GPU gets its speed from running a large number of threads at the same time; one GPU can run tens of thousands of threads at once. With n = 100, one thread per element needs 10000 threads, while one thread per row has only 100, leaving most of the GPU's compute idle. So when splitting the work, make sure there are enough threads to fill the GPU. The thread count depends on what you split by: by rows it equals the number of rows, by columns the number of columns. Which split is faster when the thread counts are the same also depends on how memory is read, and is compared in the _Tuning Kernel Performance_ chapter.

## Measured: How Long Matrix Multiplication Takes

When the previous chapter measured vector addition, the whole GPU run took about as long as the CPU, almost all of the time went into moving data, and the kernel accounted for only 0.5%. What about matrix multiplication? We measured it on an H100 as well: n = 2048, about 4.19 million elements (16 MB) per matrix. The CPU runs the three nested loops above on one core; the GPU goes through all five steps, with this section's `matMulKernel` as the kernel and 16 × 16 blocks. The GPU was warmed up once and then run 7 times, taking the median; each CPU run takes about 45 seconds, and the median of 3 runs is used.

| What                                   |    Time | Share of GPU run |
| -------------------------------------- | ------: | ---------------: |
| CPU, one core, the three nested loops  |  44.8 s |                — |
| GPU, all five steps                    |  9.0 ms |             100% |
| ├ copy to device                       | 3.62 ms |              40% |
| ├ copy back to host                    | 1.36 ms |              15% |
| ├ the kernel itself                    | 3.20 ms |              36% |
| └ allocating and freeing device memory | 0.70 ms |               8% |

The CPU and GPU times differ by a factor of about 5000, but that factor does not say much: the CPU in this comparison uses a single core running a simple, unoptimized loop, and with multiple cores and an optimized math library it would be much faster. What deserves attention is how the GPU time breaks down: copying still takes more than half, but the kernel now takes more than a third, whereas in vector addition it took less than 1%.

Both need the same back-and-forth copying; the difference comes from the ratio of data to computation. Matrix multiplication copies M and N to device memory and P back to host memory; each of the three matrices has n² elements, so the amount of data is proportional to n². The computation is n multiply-adds for each of the n² elements of P, n³ in total, a factor of n more than the data. Seen another way: an element of M is copied once but used once by each of the n threads handling the same row of P, so with n = 2048 it is copied once and used 2048 times; in vector addition, each element is used only once. The more times each element is used after being copied once, the smaller the share taken by copying.

The measurements agree: with n = 1024, the whole GPU run takes 1.91 ms, of which the kernel takes 0.45 ms, or 24%; increasing n to 2048 raises the kernel's share to 36%. How much computation each element read must feed so that reading data does not hold things back is a question the _Shared Memory and Tiling_ chapter returns to, at the level of the kernel reading device memory.

## Summary

This chapter answered the question from the start: when the data is 2D or 3D, how threads are arranged, and how each thread finds the data it is responsible for.

1. Threads can be laid out in 2D and 3D: `dim3` gives the sizes of the grid and the block, with x for columns, y for rows and z for planes.
2. Each thread's column, row and plane all equal "block index × block size + thread index", using the `.x`, `.y` and `.z` components respectively. The host rounds up separately in each direction, and the kernel checks bounds separately in each direction.
3. Memory is one-dimensional, so multidimensional data has to be flattened. In row-major order, the 2D index is `row * width + col`, and 3D adds `plane * width * height`; going from an index back to row and column uses `/` and `%`.
4. The grayscale, blur and matrix multiplication kernels each have one thread compute one output element, with more work per thread each time. Loops whose iterations are independent can become threads; loops that accumulate into the same value stay inside the thread.
5. One thread per element is not the only way to split the work. The split determines the thread count, and there must be enough threads to fill the GPU.

**CUDA only gives each thread its indices; how those indices map to the data is up to whoever writes the kernel.** When the mapping is wrong, threads still read and write, they just get the wrong data, and the program often reports no error.

This chapter did not cover how to choose the block size, or how all these threads are executed on the hardware; those are the subject of the _How a GPU Runs Your Threads_ chapter.

Further reading: Chapter 3 of _Programming Massively Parallel Processors_ (5th ed.).
