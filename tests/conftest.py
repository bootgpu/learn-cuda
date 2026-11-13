import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
CHAPTER = """\
title: { en: Basics, zh: 基础 }
stages:
  - slug: map
    title: { en: Map, zh: 映射 }
    status: draft
    difficulty: easy
  - slug: later
    title: { en: Later, zh: 以后 }
    status: wip
"""


@pytest.fixture
def bank(tmp_path: Path) -> Path:
    """A one-chapter course with a valid problem 'map'; returns the problem dir."""
    (tmp_path / "bootgpu.toml").write_text('kind = "learn"\nruntime = "cuda@1"\n')
    (tmp_path / "course.yml").write_text(
        "slug: learn-demo\nparts:\n  - title: { en: Part, zh: 篇 }\n    chapters: [basics]\n"
    )
    (tmp_path / "chapters/basics").mkdir(parents=True)
    (tmp_path / "chapters/basics/chapter.yml").write_text(CHAPTER)
    p = tmp_path / "problems/map"
    (p / "starter/cuda").mkdir(parents=True)
    shutil.copyfile(FIXTURES / "map/problem.py", p / "problem.py")
    (p / "starter/cuda/solution.cu").write_text('extern "C" void solve(const float* x, float* y, int N) {}\n')
    (p / "problem.md").write_text("")
    (p / "problem.zh.md").write_text("")
    (p / "learn.md").write_text("TODO\n")
    (p / "learn.zh.md").write_text("TODO\n")
    return p


@pytest.fixture
def sol(tmp_path: Path) -> Path:
    """map's directory in course-solutions: the official solution and a correct variant."""
    s = tmp_path / "course-solutions/learn-demo/problems/map"
    (s / "solution/cuda").mkdir(parents=True)
    (s / "solution/cuda/solution.cu").write_text("ok\n")
    (s / "variants/cuda").mkdir(parents=True)
    (s / "variants/cuda/correct_a.cu").write_text("ok\n")
    return s


QUIZ = """\
questions:
  - id: threads
    type: single
    prompt: How many threads does `kernel<<<2, 4>>>()` launch?
    options: [2, 4, 8, 6]
    answer: 2
    explanation: Two blocks of four threads.
    feedback: { 0: That is the number of blocks. }
  - id: global-index
    type: multiple
    prompt: Which expressions give a thread's global index in a 1D grid?
    options:
      - "`blockIdx.x * blockDim.x + threadIdx.x`"
      - "`threadIdx.x`"
      - "`threadIdx.x + blockIdx.x * blockDim.x`"
    answers: [2, 0]
    explanation: Both orders of the same sum.
  - id: block-size
    type: blank
    prompt: Threads per block in `kernel<<<2, 4>>>()`?
    accept: { numeric: 4 }
    explanation: The second launch parameter.
  - id: qualifier
    type: blank
    prompt: The qualifier that marks a kernel.
    accept: { exact: [__global__] }
    explanation: Kernels are `__global__` functions.
"""
QUIZ_ZH = """\
questions:
  - id: threads
    prompt: "`kernel<<<2, 4>>>()` 启动多少个线程？"
    options: [2, 4, 8, 6]
    explanation: 两个块，每块四个线程。
    feedback: { 0: 这是块数。 }
  - id: global-index
    prompt: 一维网格里，哪些表达式给出线程的全局下标？
    options:
      - "`blockIdx.x * blockDim.x + threadIdx.x`"
      - "`threadIdx.x`"
      - "`threadIdx.x + blockIdx.x * blockDim.x`"
    explanation: 同一个和，两种写法。
  - id: block-size
    prompt: "`kernel<<<2, 4>>>()` 每块几个线程？"
    explanation: 第二个启动参数。
  - id: qualifier
    prompt: 标记核函数的限定符。
    explanation: 核函数是 `__global__` 函数。
"""


@pytest.fixture
def quiz(bank: Path) -> Path:
    """A valid quiz stage 'check' (draft) beside 'map'; returns its dir."""
    root = bank.parent.parent
    chapter = root / "chapters/basics/chapter.yml"
    chapter.write_text(
        chapter.read_text() + "  - slug: check\n    title: { en: Check, zh: 小测 }\n    status: draft\n"
    )
    q = root / "problems/check"
    q.mkdir()
    (q / "quiz.yml").write_text(QUIZ)
    (q / "quiz.zh.yml").write_text(QUIZ_ZH)
    (q / "learn.md").write_text("TODO\n")
    (q / "learn.zh.md").write_text("TODO\n")
    return q
