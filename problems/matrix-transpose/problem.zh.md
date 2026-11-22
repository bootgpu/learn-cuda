写一个 GPU 程序，把 `rows` 行、`cols` 列的 32 位浮点数矩阵 `input` 转置，结果存入 `output`。
转置就是行列互换：`input` 第 r 行第 c 列的元素，放到 `output` 第 c 行第 r 列，所以 `output` 有
`cols` 行、`rows` 列。两个矩阵都按行优先存储。

::motion{path="learn-cuda/matrix-transpose"}

每个线程处理一个元素。`solve` 里 block 已经定为 16 × 16，grid 由你来算。

## 实现要求

- 不允许使用外部库
- 补全 `transposeKernel`，名字和参数保持不变
- 补全 `solve` 里的 `grid`，让线程覆盖每一个元素；`solve` 函数的签名必须保持不变
- 最终结果必须存放在 `output` 中
- 不得修改输入，也不得读写给定数组范围以外的内存

## 示例 1：

```
输入：rows = 2, cols = 4
      input = [[0.5, -1.0,  2.5, 4.0],
               [3.0,  7.5, -2.0, 6.0]]
输出：output = [[ 0.5,  3.0],
                [-1.0,  7.5],
                [ 2.5, -2.0],
                [ 4.0,  6.0]]
```

## 示例 2：

```
输入：rows = 1, cols = 3
      input = [[9.0, 8.5, -3.0]]
输出：output = [[ 9.0],
                [ 8.5],
                [-3.0]]
```

## 约束

- 1 ≤ `rows`, `cols`，且 `rows` × `cols` ≤ 10,000
- `input` 和 `output` 各含 `rows` × `cols` 个连续的 32 位浮点数，位于 GPU 显存
- -100 ≤ 矩阵元素 ≤ 100
