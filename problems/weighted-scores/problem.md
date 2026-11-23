Write a GPU program that computes each student's weighted total. `scores` is a grade table of
`students` rows and `subjects` columns: row i holds student i's score in each subject, stored in
row-major order. `weights` holds one weight for each of the `subjects` subjects. Student i's total is
each score times that subject's weight, summed, and goes in `totals[i]`:

```
totals[i] = scores[i][0] × weights[0]
          + scores[i][1] × weights[1]
          + …
          + scores[i][subjects - 1] × weights[subjects - 1]
```

::motion{path="learn-cuda/weighted-scores"}

All numbers are 32-bit floats.

The `solve` function is fixed (see `harness.cu`): it launches one-dimensional threads by the number
of students, 256 per block, with the grid rounded up. Each thread computes one student's total.

## Implementation Requirements

- External libraries are not permitted
- Implement `weightedScoresKernel`, keeping its name and parameters
- Do not write `solve`; the judge supplies it
- The final result must be stored in `totals`
- Do not modify the inputs or access memory outside the given arrays

## Example 1:

```
Input:  students = 2, subjects = 3
        scores = [[80.0,  90.0, 70.0],
                  [60.0, 100.0, 50.0]]
        weights = [0.3, 0.5, 0.2]
Output: totals = [83.0, 78.0]
```

## Example 2:

```
Input:  students = 3, subjects = 2
        scores = [[ 90.0, 80.0],
                  [ 70.0, 60.0],
                  [100.0, 40.0]]
        weights = [0.5, 0.5]
Output: totals = [85.0, 65.0, 70.0]
```

## Constraints

- 1 ≤ `students`, `subjects`, and `students` × `subjects` ≤ 10,000
- `scores` holds `students` × `subjects`, `weights` holds `subjects` and `totals` holds `students`
  contiguous 32-bit floats, all in GPU memory
- 0 ≤ scores ≤ 100, 0 ≤ weights ≤ 1
