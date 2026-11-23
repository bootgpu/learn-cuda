写一个 GPU 程序，算出每名学生的加权总分。`scores` 是 `students` 行、`subjects` 列的成绩表，
第 i 行是第 i 名学生各科的分数，按行优先存储；`weights` 是 `subjects` 个科目权重。第 i 名学生的总分是
他每科分数乘以该科权重再相加，结果存入 `totals[i]`：

```
totals[i] = scores[i][0] × weights[0]
          + scores[i][1] × weights[1]
          + …
          + scores[i][subjects - 1] × weights[subjects - 1]
```

::motion{path="learn-cuda/weighted-scores"}

所有数都是 32 位浮点数。

`solve` 是固定的（见 `harness.cu`）：按学生数启动一维的线程，每个 block 256 个，grid 向上取整。
每个线程算一名学生的总分。

## 实现要求

- 不允许使用外部库
- 补全 `weightedScoresKernel`，名字和参数保持不变
- 不要写 `solve`，判题时会提供
- 最终结果必须存放在 `totals` 中
- 不得修改输入，也不得读写给定数组范围以外的内存

## 示例 1：

```
输入：students = 2, subjects = 3
      scores = [[80.0,  90.0, 70.0],
                [60.0, 100.0, 50.0]]
      weights = [0.3, 0.5, 0.2]
输出：totals = [83.0, 78.0]
```

## 示例 2：

```
输入：students = 3, subjects = 2
      scores = [[ 90.0, 80.0],
                [ 70.0, 60.0],
                [100.0, 40.0]]
      weights = [0.5, 0.5]
输出：totals = [85.0, 65.0, 70.0]
```

## 约束

- 1 ≤ `students`, `subjects`，且 `students` × `subjects` ≤ 10,000
- `scores` 含 `students` × `subjects` 个、`weights` 含 `subjects` 个、`totals` 含 `students` 个连续的
  32 位浮点数，都位于 GPU 显存
- 0 ≤ 分数 ≤ 100，0 ≤ 权重 ≤ 1
