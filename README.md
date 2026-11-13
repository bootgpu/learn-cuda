# learn-cuda

Problems for bootgpu's CUDA Parallel Programming course: write CUDA C++ kernels chapter by chapter, from GPU
programming foundations to classic parallel algorithms. Tests stay small
(up to about 10,000 elements), so any correct solution passes; speed is for a later course.

## Layout

```text
course.yml, chapters/              course outline: chapters and their stages, in order
chapters/<chapter>/
  reading.md, reading.zh.md        the chapter's study notes (English / Chinese); required
                                   once a stage is published, no '# ' title; TODO is fine for now
problems/<slug>/
  problem.md, problem.zh.md        statement (English / Chinese)
  problem.py                       reference implementation, signature and test cases
  starter/cuda/solution.cu         starter code
  harness.cu                       optional: a fixed solve() that includes solution.cu; the judge compiles it instead
bootgpu.toml                       runtime, GPUs and time limit
scripts/                           validate.py, verify.py, build.py
archive/                           the earlier learn-cuda-basics outline and problems, not published
```

## Checks

```sh
uv sync
uv run scripts/validate.py   # course layout and every problem, on CPU
uv run pytest -q
```

## License

[CC BY-NC-ND 4.0](LICENSE).
