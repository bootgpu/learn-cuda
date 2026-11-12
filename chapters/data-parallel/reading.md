## Celsius to Fahrenheit

Every thread runs the same kernel. A thread finds its own element from its block number and its thread number:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

`blockIdx.x * blockDim.x` counts the positions the blocks before this one have already taken; `threadIdx.x` is the thread's place inside its own block.

`solve` rounds the number of blocks up, `(n + threads - 1) / threads`, so no element is left without a thread. The last block may then have threads whose `i` is past the end of the array. Those threads must not touch memory, so guard the work with `if (i < n)`.

One C detail: `9 / 5` between two integers is `1`, not `1.8`. `celsius[i] * 9 / 5` is fine, since `celsius[i]` is a float and the multiplication comes first.

## Two Elements per Thread

This stage practices the index formula above under a new launch.

With one element per thread, the formula reads "positions taken by the blocks before this one, plus my place in my block". Ask the same two questions again for this launch. How many positions does each earlier block take now that every thread handles two elements? And where inside its block does a thread's pair start? Once a thread knows its first element, the second one needs its own check against `n`: with an odd `n`, the last thread has only one element to compute.

Taking two neighbouring elements is not the only way. A thread could also take one element in the first half of the array and one in the second half. Both pass here; the _Tuning Kernel Performance_ chapter (memory coalescing) shows why the choice matters for speed.
