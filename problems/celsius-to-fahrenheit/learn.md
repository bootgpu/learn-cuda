Every thread runs the same kernel. A thread finds its own element from its block number
and its thread number:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

`blockIdx.x * blockDim.x` counts the positions the blocks before this one have already
taken; `threadIdx.x` is the thread's place inside its own block.

`solve` rounds the number of blocks up, `(n + threads - 1) / threads`, so no element is
left without a thread. The last block may then have threads whose `i` is past the end of
the array. Those threads must not touch memory, so guard the work with `if (i < n)`.

One C detail: `9 / 5` between two integers is `1`, not `1.8`. `celsius[i] * 9 / 5` is
fine, since `celsius[i]` is a float and the multiplication comes first.
