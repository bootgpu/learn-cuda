Given an array `x` of `N` floats, write a kernel that computes

```
y[i] = 2 * x[i] + 1
```

for every `i`, with one thread per element.

## Requirements

- Implement `solve(const float* x, float* y, int N)` in `solution.cu`. Keep its signature and `extern "C"` as in the starter.
- `x` and `y` already point to GPU memory. Launch your kernel from `solve`.
- Write the result into `y`; its contents on entry are undefined.

## Examples

```
Input:  x = [1.0, 2.0, 3.0, 4.0], N = 4
Output: y = [3.0, 5.0, 7.0, 9.0]
```

```
Input:  x = [-0.5, 0.0, 10.0], N = 3
Output: y = [0.0, 1.0, 21.0]
```

## Constraints

- 1 ≤ N ≤ 1024, so one block of `N` threads is enough
- -100 ≤ x[i] ≤ 100
- An output is accepted if it is within 1e-5 + 1e-5 × |expected| of the expected value
