Write a GPU program that fades out a grayscale video, darkening it bit by bit, and stores the result in
`output`. The video `input` has `frames` frames of `height` rows and `width` columns, and each pixel
is a brightness stored as a 32-bit float. Every pixel of frame t (counting from 0) is multiplied by
`(frames - t) / frames`: frame 0 stays as it is, and each later frame is darker.

::motion{path="learn-cuda/video-fade"}

Both `input` and `output` are stored frame after frame, each frame in row-major order.

The `solve` function is fixed (see `harness.cu`): the block is 8 × 8 × 4 and the grid is rounded
up in all three directions; x is the column, y the row and z the frame. Each thread handles one
pixel.

## Implementation Requirements

- External libraries are not permitted
- Implement `videoFadeKernel`, keeping its name and parameters
- Do not write `solve`; the judge supplies it
- The final result must be stored in `output`
- Do not modify the input or access memory outside the given arrays

## Example 1:

```
Input:  frames = 2, height = 2, width = 3
        input = [[[0.2, 0.4, 0.6],
                  [0.8, 1.0, 0.0]],     frame 0
                 [[1.0, 0.6, 0.2],
                  [0.0, 0.4, 0.8]]]     frame 1
Output: output = [[[0.2, 0.4, 0.6],
                   [0.8, 1.0, 0.0]],    frame 0 × 1
                  [[0.5, 0.3, 0.1],
                   [0.0, 0.2, 0.4]]]    frame 1 × 0.5
```

## Example 2:

```
Input:  frames = 4, height = 1, width = 2
        input = [[[1.0, 0.4]], [[1.0, 0.4]], [[1.0, 0.4]], [[1.0, 0.4]]]
Output: output = [[[1.0, 0.4]], [[0.75, 0.3]], [[0.5, 0.2]], [[0.25, 0.1]]]
```

## Constraints

- 1 ≤ `frames`, `height`, `width`, and `frames` × `height` × `width` ≤ 10,000
- `input` and `output` each hold `frames` × `height` × `width` contiguous 32-bit floats in GPU
  memory
- 0 ≤ pixel brightness ≤ 1
