# learn-cuda

English | [中文](README.zh.md)

The problems of **CUDA Parallel Programming**, a [BootGPU](https://www.bootgpu.com) course: write CUDA C++ kernels chapter by chapter, from GPU programming foundations to classic parallel algorithms.

To take the course, open the [course page](https://www.bootgpu.com/courses/learn-cuda/). Each chapter has study notes and a few stages, quizzes or coding problems; for a coding problem you write the kernel in the browser and submit it to be graded on an NVIDIA T4. Tests stay small (up to about 10,000 elements), so any correct kernel passes; making kernels fast is the subject of a later course, Peak LLM Kernels. The official solutions are not in this repository; the site shows them once you pass a stage. Chapters are published as they are finished.

## For authors

### Layout

```text
course.yml                         course outline: parts and their chapters, in order
chapters/<chapter>/
  chapter.yml                      the chapter's title and its stages, each with a status
  reading.md, reading.zh.md        the chapter's study notes (English / Chinese)
problems/<slug>/                   a coding stage
  problem.md, problem.zh.md        statement (English / Chinese)
  problem.py                       reference implementation, signature and test cases
  starter/cuda/solution.cu         starter code
  harness.cu                       optional: a fixed solve() that includes solution.cu; the judge compiles it instead
problems/<slug>/                   a quiz stage
  quiz.yml, quiz.zh.yml            questions (English / Chinese)
bootgpu/problem.py                 the base class of every problem.py
bootgpu.toml                       runtime, GPUs and time limit
scripts/                           validate.py, verify.py, build.py
tests/                             tests of the scripts
archive/                           an earlier outline and retired problems; not on the site
```

### Checks

```sh
uv sync
uv run scripts/validate.py   # course layout and every problem, on CPU
uv run pytest -q
```

`scripts/verify.py` grades the official solutions on the GPU; it needs the private solutions repository and judge keys.

## License

[CC BY-NC-ND 4.0](LICENSE): you may share this material with attribution, but not use it commercially or distribute modified versions.
