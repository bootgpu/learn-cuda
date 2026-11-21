Write a GPU program that computes the product P = M × N of two `n` × `n` square matrices and stores
it in `P`. `M`, `N` and `P` are all stored in row-major order. The element in row `row`, column `col`
of P is row `row` of M times column `col` of N, element by element, summed:

```
P[row][col] = M[row][0] × N[0][col]
            + M[row][1] × N[1][col]
            + …
            + M[row][n - 1] × N[n - 1][col]
```

All numbers are 32-bit floats.

The `solve` function is fixed (see `harness.cu`): it launches `n` one-dimensional threads in total,
32 per block, with the grid rounded up. Each thread computes a whole row of P: thread `row`
(counting from 0) computes all `n` elements of row `row`.

## Implementation Requirements

- External libraries are not permitted
- Implement `matMulRowKernel`, keeping its name and parameters
- Do not write `solve`; the judge supplies it
- The final result must be stored in `P`
- Do not modify the inputs or access memory outside the given arrays

## Example 1:

```
Input:  n = 2
        M = [[1.0, 2.0],
             [3.0, 4.0]]
        N = [[2.0, 0.0],
             [1.0, 3.0]]
Output: P = [[ 4.0,  6.0],
             [10.0, 12.0]]
```

## Example 2:

```
Input:  n = 3
        M = [[1.0, 2.0, 0.0],
             [0.0, 1.0, 1.0],
             [2.0, 0.0, 1.0]]
        N = [[1.0, 0.0, 1.0],
             [0.0, 2.0, 0.0],
             [1.0, 1.0, 0.0]]
Output: P = [[1.0, 4.0, 1.0],
             [1.0, 3.0, 0.0],
             [3.0, 1.0, 2.0]]
```

## Constraints

- 1 ≤ `n` ≤ 100
- `M`, `N` and `P` each hold `n` × `n` contiguous 32-bit floats, all in GPU memory
- −10 ≤ elements of `M` and `N` ≤ 10
