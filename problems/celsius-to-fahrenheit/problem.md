Given `n` temperatures in degrees Celsius, write a kernel that converts each one to
Fahrenheit:

```
fahrenheit[i] = celsius[i] * 9 / 5 + 32
```

with one thread per element.

## Requirements

- Fill in `celsiusToFahrenheitKernel` in `solution.cu`. The starter's `solve` already
  launches it with 256 threads per block and enough blocks to cover `n`.
- Keep the signature of `solve` and its `extern "C"`.
- `celsius` and `fahrenheit` already point to GPU memory.
- Write the result into `fahrenheit`; its contents on entry are undefined.

## Examples

```
Input:  celsius = [0.0, 100.0, -40.0, 37.0], n = 4
Output: fahrenheit = [32.0, 212.0, -40.0, 98.6]
```

```
Input:  celsius = [20.0, 25.5, -10.0], n = 3
Output: fahrenheit = [68.0, 77.9, 14.0]
```

## Constraints

- 1 ≤ n ≤ 10,000; `n` need not be a multiple of 256
- -100 ≤ celsius[i] ≤ 100
- An output is accepted if it is within 1e-5 + 1e-5 × |expected| of the expected value
