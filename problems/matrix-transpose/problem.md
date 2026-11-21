Write a GPU program that transposes `input`, a matrix of 32-bit floats with `rows` rows and `cols`
columns, and stores the result in `output`. Transposing swaps rows and columns: the element in row r,
column c of `input` goes to row c, column r of `output`, so `output` has `cols` rows and `rows`
columns. Both matrices are stored in row-major order.

Each thread handles one element. `solve` already sets the block to 16 × 16; working out the grid is
up to you.

## Implementation Requirements

- External libraries are not permitted
- Implement `transposeKernel`, keeping its name and parameters
- Fill in `grid` in `solve` so that the threads cover every element; the signature of `solve` must
  remain unchanged
- The final result must be stored in `output`
- Do not modify the input or access memory outside the given arrays

## Example 1:

```
Input:  rows = 2, cols = 4
        input = [[0.5, -1.0,  2.5, 4.0],
                 [3.0,  7.5, -2.0, 6.0]]
Output: output = [[ 0.5,  3.0],
                  [-1.0,  7.5],
                  [ 2.5, -2.0],
                  [ 4.0,  6.0]]
```

## Example 2:

```
Input:  rows = 1, cols = 3
        input = [[9.0, 8.5, -3.0]]
Output: output = [[ 9.0],
                  [ 8.5],
                  [-3.0]]
```

## Constraints

- 1 ≤ `rows`, `cols`, and `rows` × `cols` ≤ 10,000
- `input` and `output` each hold `rows` × `cols` contiguous 32-bit floats in GPU memory
- -100 ≤ matrix elements ≤ 100
