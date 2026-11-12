Given two arrays `a` and `b` of `n` floats, compute

```
c[i] = a[i] + b[i]
```

for every `i`, with each thread computing two elements.

In this stage `solve` is fixed. It launches 256 threads per block, but only half as many
threads in total as there are elements:

```cpp
extern "C" void solve(const float* a, const float* b, float* c, int n) {
    int threads = 256;
    int blocks = (n + 2 * threads - 1) / (2 * threads);
    vecAddKernel<<<blocks, threads>>>(a, b, c, n);
}
```

So each block covers 512 elements, and every thread has to handle two of them. Which two
is up to you, as long as the threads together cover every element.

## Requirements

- Write `vecAddKernel` in `solution.cu` and keep its name and parameters.
- Do not write `solve`: the judge adds the one above, and a second `solve` fails to
  compile.
- `a`, `b` and `c` already point to GPU memory.
- Write the result into `c`; its contents on entry are undefined.

## Examples

```
Input:  a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [10.0, 20.0, 30.0, 40.0, 50.0], n = 5
Output: c = [11.0, 22.0, 33.0, 44.0, 55.0]
```

```
Input:  a = [0.5, -1.5, 2.0, 3.0]
        b = [0.5, 1.5, -4.0, 7.0], n = 4
Output: c = [1.0, 0.0, -2.0, 10.0]
```

## Constraints

- 1 ≤ n ≤ 10,000; `n` may be odd
- -100 ≤ a[i], b[i] ≤ 100
- An output is accepted if it is within 1e-5 + 1e-5 × |expected| of the expected value
