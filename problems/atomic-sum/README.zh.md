给定 `N` 个 float 组成的数组 `x`，求它们的和，存进 `sum[0]`。`N` 可能远大于一个 block 能处理的量，所以要启动多个 block，每个 block 用 `atomicAdd` 把自己的部分和加到 `sum[0]` 上。

## 要求

- 在 `solution.cu` 里实现 `solve(const float* x, float* sum, int N)`，签名和 `extern "C"` 保持初始代码的样子。
- `x` 和 `sum` 已经指向 GPU 显存；`sum` 只有一个 float。
- 进入 `solve` 时 `sum[0]` 的值是未定义的。要在任何 block 往上加之前把它置 0，例如在 `solve` 里启动 kernel 之前调用 `cudaMemset`。在同一个 kernel 里置 0 不行：别的 block 可能已经加过了。

## 示例

```
输入：x = [1.0, 2.0, 3.0, 4.0], N = 4
输出：sum = [10.0]
```

```
输入：x = [0.5, -1.5, 2.0], N = 3
输出：sum = [1.0]
```

## 约束

- 1 ≤ N ≤ 10,000
- -10 ≤ x[i] ≤ 10
- 各 block 相加的先后不固定，结果的末几位每次运行可能不同。输出与期望值之差不超过 1e-3 + 1e-4 × |期望值| 即判对
