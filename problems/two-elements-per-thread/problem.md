Write a GPU program that adds two arrays `a` and `b` of `n` 32-bit floats element-wise and
stores the sums in `c`. The `solve` function is fixed (see `harness.cu`): it launches one thread
for every two elements, so each thread computes two elements. Which two is up
to you, as long as every element is covered.

## Implementation Requirements

- External libraries are not permitted
- Implement `vecAddKernel`, keeping its name and parameters
- Do not write `solve`; the judge supplies it
- The final result must be stored in `c`
- Do not modify the inputs or access memory outside the given arrays

## Example 1:

```
Input:  a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [10.0, 20.0, 30.0, 40.0, 50.0]
Output: c = [11.0, 22.0, 33.0, 44.0, 55.0]
```

## Example 2:

```
Input:  a = [0.5, -1.5, 2.0, 3.0]
        b = [0.5, 1.5, -4.0, 7.0]
Output: c = [1.0, 0.0, -2.0, 10.0]
```

## Constraints

- 1 ≤ `n` ≤ 10,000; `n` may be odd
- Threads are launched in whole blocks of 256, so there may be more than n/2 of them; check both
  elements against the bounds
- `a`, `b` and `c` each hold `n` contiguous 32-bit floats in GPU memory
- -100 ≤ `a[i]`, `b[i]` ≤ 100
