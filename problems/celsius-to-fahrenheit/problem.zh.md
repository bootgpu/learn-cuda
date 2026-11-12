给定 `n` 个摄氏温度，写一个 kernel 把每个都换算成华氏温度：

```
fahrenheit[i] = celsius[i] * 9 / 5 + 32
```

每个线程负责一个元素。

## 要求

- 在 `solution.cu` 里补全 `celsiusToFahrenheitKernel`。初始代码里的 `solve` 已经用
  每个 block 256 个线程、足够覆盖 `n` 的 block 数启动了它。
- `solve` 的签名和 `extern "C"` 保持不变。
- `celsius` 和 `fahrenheit` 已经指向 GPU 显存。
- 结果写进 `fahrenheit`；进入 `solve` 时它里面的内容是未定义的。

## 示例

```
输入：celsius = [0.0, 100.0, -40.0, 37.0], n = 4
输出：fahrenheit = [32.0, 212.0, -40.0, 98.6]
```

```
输入：celsius = [20.0, 25.5, -10.0], n = 3
输出：fahrenheit = [68.0, 77.9, 14.0]
```

## 约束

- 1 ≤ n ≤ 10,000；`n` 不一定是 256 的倍数
- -100 ≤ celsius[i] ≤ 100
- 输出与期望值之差不超过 1e-5 + 1e-5 × |期望值| 即判对
