::bilibili[第 2 章视频：从循环到 kernel]{bvid="BV1nkW962EJS"}

所有线程跑的是同一个 kernel。每个线程靠自己的 block 编号和线程编号，算出该处理哪个元素：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

`blockIdx.x * blockDim.x` 是前面的 block 已经占掉的位置数，`threadIdx.x` 是这个线程在自己
block 里的位置。

`solve` 里的 block 数是向上取整的：`(n + threads - 1) / threads`，保证每个元素都有线程管。
代价是最后一个 block 里可能有线程的 `i` 超出了数组末尾，这些线程不能碰内存，所以要用
`if (i < n)` 把计算包起来。

C 语言的一个细节：两个整数相除，`9 / 5` 等于 `1`，不是 `1.8`。写成
`celsius[i] * 9 / 5` 没问题：先算乘法，`celsius[i]` 是 float，结果也是 float。
