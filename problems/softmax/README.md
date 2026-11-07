Given an array `x` of `N` floats, compute its softmax into `y`:

```
m    = max_j x[j]
y[i] = exp(x[i] - m) / sum_j exp(x[j] - m)
```

Subtracting the maximum `m` does not change the result, but it keeps `exp` from overflowing when `x` holds large values.

## Requirements

- Implement `solve(const float* x, float* y, int N)` in `solution.cu`. Keep its signature and `extern "C"` as in the starter.
- `x` and `y` already point to GPU memory; the contents of `y` on entry are undefined.
- `m` and the sum each depend on all of `x`, so one kernel launch is not enough: launch several from `solve` (for example: maximum, then sum, then division). Allocate any scratch memory with `cudaMalloc` inside `solve`.

## Examples

```
Input:  x = [1.0, 2.0, 3.0], N = 3
Output: y = [0.0900, 0.2447, 0.6652]
```

```
Input:  x = [1000.0, 1000.0], N = 2
Output: y = [0.5, 0.5]
```

## Constraints

- 1 ≤ N ≤ 10,000
- -100 ≤ x[i] ≤ 100
- An output is accepted if it is within 1e-9 + 1e-4 × |expected| of the expected value
