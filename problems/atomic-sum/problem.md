Given an array `x` of `N` floats, compute their sum and store it in `sum[0]`. `N` can be far larger than one block, so launch many blocks and let each block add its partial sum to `sum[0]` with `atomicAdd`.

## Requirements

- Implement `solve(const float* x, float* sum, int N)` in `solution.cu`. Keep its signature and `extern "C"` as in the starter.
- `x` and `sum` already point to GPU memory; `sum` holds one float.
- The value in `sum[0]` on entry is undefined. Set it to 0 before any block adds to it, for example with `cudaMemset` in `solve` before the launch. Zeroing it inside the same kernel does not work: other blocks may already have added their part.

## Examples

```
Input:  x = [1.0, 2.0, 3.0, 4.0], N = 4
Output: sum = [10.0]
```

```
Input:  x = [0.5, -1.5, 2.0], N = 3
Output: sum = [1.0]
```

## Constraints

- 1 ≤ N ≤ 10,000
- -10 ≤ x[i] ≤ 10
- Blocks add in any order, so the last digits of the result can change from run to run. An output is accepted if it is within 1e-3 + 1e-4 × |expected| of the expected value
