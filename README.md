# learn-cuda-basics

Problems for bootgpu's CUDA Basics course: write your first CUDA C++ kernels and get them right. Tests stay small
(up to about 10,000 elements), so any correct solution passes; speed is for a later course. The format is inspired
by [LeetGPU](https://github.com/AlphaGPU/leetgpu-challenges)'s challenges.

## Layout

```text
course.yml, chapters/              course outline: chapters and their stages, in order
problems/<slug>/
  problem.md, problem.zh.md        statement (English / Chinese)
  learn.md, learn.zh.md            study notes (English / Chinese); TODO is fine for now
  problem.py                       reference implementation, signature and test cases
  starter/cuda/solution.cu         starter code
bootgpu.toml                       runtime, GPUs and time limit
scripts/                           validate.py, verify.py, build.py
```

## Checks

```sh
uv sync
uv run scripts/validate.py   # course layout and every problem, on CPU
uv run pytest -q
```

## License

[CC BY-NC-ND 4.0](LICENSE).
