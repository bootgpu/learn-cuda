Write a GPU program that converts `n` temperatures from Celsius to Fahrenheit. The inputs and
outputs are 32-bit floats, and each thread converts one element:

```
fahrenheit[i] = celsius[i] * 9 / 5 + 32
```

## Implementation Requirements

- External libraries are not permitted
- Implement `celsiusToFahrenheitKernel`; the given `solve` already launches it
- The `solve` function signature must remain unchanged
- The final result must be stored in `fahrenheit`
- Do not modify the input or access memory outside the given arrays

## Example 1:

```
Input:  celsius = [0.0, 100.0, -40.0, 37.0]
Output: fahrenheit = [32.0, 212.0, -40.0, 98.6]
```

## Example 2:

```
Input:  celsius = [20.0, 25.5, -10.0]
Output: fahrenheit = [68.0, 77.9, 14.0]
```

## Constraints

- 1 ≤ `n` ≤ 10,000
- `celsius` and `fahrenheit` each hold `n` contiguous 32-bit floats in GPU memory
- -100 ≤ `celsius[i]` ≤ 100
