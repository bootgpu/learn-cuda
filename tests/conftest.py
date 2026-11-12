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
    (p / "problem.yml").write_text("title:\n  en: Map\n  zh: 映射\ndifficulty: easy\n")
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
