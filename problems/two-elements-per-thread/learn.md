This stage practices the index formula from Chapter 2 of PMPP under a new launch.

With one element per thread, the formula reads "positions taken by the blocks before
this one, plus my place in my block":

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

Ask the same two questions again for this launch. How many positions does each earlier
block take now that every thread handles two elements? And where inside its block does a
thread's pair start? Once a thread knows its first element, the second one needs its own
check against `n`: with an odd `n`, the last thread has only one element to compute.

Taking two neighbouring elements is not the only way. A thread could also take one
element in the first half of the array and one in the second half. Both pass here;
Chapter 6 (memory coalescing) shows why the choice matters for speed.
