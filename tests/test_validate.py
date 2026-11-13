import shutil

import validate


def test_valid(bank):
    assert validate.validate(bank) == []
    assert validate.validate_course(bank.parent.parent) == []


def test_main_exit_codes(bank, capsys):
    assert validate.main([str(bank)]) == 0
    (bank / "variants/cuda").mkdir(parents=True)
    assert validate.main([str(bank)]) == 1
    assert "map/variants/: belongs in bootgpu/course-solutions" in capsys.readouterr().err


def test_errors_are_collected(bank):
    src = (bank / "problem.py").read_text()
    src = src.replace("import torch\n", "import torch\nimport numpy\nfrom . import helper\n")
    src = src.replace("def examples(self):\n", "def examples(self):\n        torch.manual_seed(1)\n")
    (bank / "problem.py").write_text(src)
    (bank / "starter/cuda/solution.cu").unlink()
    (bank / "solution/cuda").mkdir(parents=True)
    (bank / "solution/cuda/solution.cu").write_text("")
    errors = validate.validate(bank)
    assert len(errors) == 5, errors
    for part in ["'numpy'", "'.'", "manual_seed()", "starter/cuda/solution.cu: missing", "solution/: "]:
        assert any(part in e for e in errors), part


def test_problem_checks(bank):
    p = bank / "problem.py"
    src = p.read_text()
    p.write_text(src.replace("return [self._case([1.0, 2.0, 3.0, 4.0])]", "return [self._rand(4)]"))
    assert any("example_1 changes with the seed" in e for e in validate.validate(bank))

    # out buffers are uninitialized memory (torch.empty) and may differ between seeds
    out = '"y": torch.empty_like(x), "N": x.numel()'
    garbage = src.replace(out, out.replace("empty_like", "rand_like"))
    assert garbage != src
    p.write_text(garbage)
    assert validate.validate(bank) == []

    p.write_text(src.replace('"N": x.numel()}', '"N": x.numel(), "extra": 1}'))
    assert any("example_1" in e and "problem.py" in e for e in validate.validate(bank))

    p.write_text(src.replace("return [self._case([1.0, 2.0, 3.0, 4.0])]", "return []"))
    assert validate.validate(bank) == ["problem.py: ValueError: examples() must return a non-empty list"]


def test_no_performance(bank):
    p = bank / "problem.py"
    p.write_text(p.read_text() + "\n    def performance(self):\n        return self._rand(4096)\n")
    assert validate.validate(bank) == [
        "problem.py: performance() is not allowed; learn courses only check correctness"
    ]


def test_harness_includes_the_solution_first(bank):
    (bank / "harness.cu").write_text('#include "solution.cu"\n\nextern "C" void solve() {}\n')
    assert validate.validate(bank) == []
    (bank / "harness.cu").write_text('#include <cuda_runtime.h>\n#include "solution.cu"\n')
    assert validate.validate(bank) == ['harness.cu: the first line must be #include "solution.cu"']


def test_kind(bank):
    root = bank.parent.parent
    (root / "bootgpu.toml").write_text('kind = "peak"\nruntime = "cuda@1"\n')
    (root / "course.yml").write_text((root / "course.yml").read_text().replace("learn-demo", "demo"))
    assert validate.validate_course(root) == [
        'bootgpu.toml: kind must be "learn"',
        "course.yml: slug must start with learn-",
    ]


def test_tensor_limit(bank):
    p = bank / "problem.py"
    p.write_text(p.read_text().replace("self._rand(10_000)", "self._rand(10_001)"))
    assert validate.validate(bank) == [
        "problem.py: test_6: x has 10,001 elements, above 10,000; learn tests stay small"
        " (max_tensor_elements on the stage in chapter.yml raises it)",
        "problem.py: test_6: y has 10,001 elements, above 10,000; learn tests stay small"
        " (max_tensor_elements on the stage in chapter.yml raises it)",
    ]

    root = bank.parent.parent
    chapter = root / "chapters/basics/chapter.yml"
    chapter.write_text(
        chapter.read_text().replace(
            "difficulty: easy\n", "difficulty: easy\n    max_tensor_elements: 20000\n"
        )
    )
    assert validate.validate(bank) == []

    chapter.write_text(chapter.read_text().replace("20000", "10000"))
    assert validate.validate_course(root) == [
        "chapters/basics/chapter.yml: stages[1]: max_tensor_elements must be an integer above 10000"
    ]


def test_bad_slug(bank):
    bad = bank.parent / "Map_1"
    shutil.move(bank, bad)
    assert validate.validate(bad)[0].startswith("slug")


def test_stage(bank):
    root = bank.parent.parent
    chapter = root / "chapters/basics/chapter.yml"
    chapter.write_text(chapter.read_text().replace("difficulty: easy", "difficulty: trivial"))
    assert validate.validate_course(root) == [
        "chapters/basics/chapter.yml: stages[1]: difficulty must be one of easy, medium, hard"
    ]
    chapter.write_text(chapter.read_text().replace("    difficulty: trivial\n", ""))
    assert validate.validate(bank) == ["the stage needs a difficulty in chapters/basics/chapter.yml"]
    chapter.write_text(
        chapter.read_text().replace("status: draft\n", "status: draft\n    difficulty: easy\n")
    )
    summary = "    summary: { en: One thread per element., zh: 一个线程算一个元素。 }\n"
    chapter.write_text(chapter.read_text().replace(summary, "    summary: { en: ' ' }\n"))
    assert validate.validate_course(root) == [
        "chapters/basics/chapter.yml: stages[1]: summary.en must be a string of 1-200 characters",
        "chapters/basics/chapter.yml: stages[1]: summary.zh must be a string of 1-200 characters",
    ]
    chapter.write_text(chapter.read_text().replace("    summary: { en: ' ' }\n", ""))
    assert validate.validate(bank) == ["the stage needs a summary in chapters/basics/chapter.yml"]
    chapter.write_text(chapter.read_text().replace("difficulty: easy\n", "difficulty: easy\n" + summary))

    later = bank.parent / "later"
    shutil.copytree(bank, later)
    assert any("the stage is wip" in e for e in validate.validate(later))
    other = bank.parent / "other"
    shutil.move(later, other)
    assert "not a stage in any chapter.yml listed by course.yml" in validate.validate(other)


def test_learn(bank):
    (bank / "learn.zh.md").unlink()
    assert validate.validate(bank) == ["learn.zh.md: missing"]


def test_course(bank):
    root = bank.parent.parent
    chapter = root / "chapters/basics/chapter.yml"
    chapter.write_text(
        chapter.read_text().replace("status: wip", "status: published").replace("- slug: map", "- slug: Map")
    )
    (root / "chapters/extra").mkdir()
    (root / "chapters/extra/chapter.yml").write_text("title: { en: X, zh: X }\nstages: []\n")
    (root / "bootgpu.toml").write_text('runtime = "cuda@1"\n')
    errors = validate.validate_course(root)
    assert len(errors) == 4, errors
    for part in [
        "chapters/extra: not listed",
        "stages[1]: slug must be",
        "stage later is published but problems/later is missing",
        "bootgpu.toml: kind",
    ]:
        assert any(part in e for e in errors), part

    (root / "course.yml").write_text(
        "parts:\n  - title: { en: P, zh: 篇 }\n    chapters: [basics, extra, basics]\n"
    )
    (root / "chapters/extra/chapter.yml").write_text(
        "title: { en: X, zh: X }\nstages:\n  - slug: later\n    title: { en: L, zh: L }\n    status: wip\n"
    )
    errors = validate.validate_course(root)
    assert "course.yml: chapter 'basics' is listed twice" in errors
    assert "stage 'later' is in chapters/basics and chapters/extra" in errors
