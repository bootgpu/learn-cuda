import build

HEADER = "title: { en: Demo, zh: 演示 }\nsummary: { en: A demo., zh: 演示课。 }\n"


def course(bank, status="published"):
    root = bank.parent.parent
    (root / "course.yml").write_text(HEADER + (root / "course.yml").read_text())
    (root / "bootgpu.toml").write_text((root / "bootgpu.toml").read_text() + 'gpus = ["t4"]\n')
    chapter = root / "chapters/basics/chapter.yml"
    chapter.write_text(chapter.read_text().replace("status: draft", f"status: {status}"))
    return root


def test_package(bank):
    root = course(bank)
    (bank / "problem.md").write_text("# Map\n")
    (bank / "learn.md").write_text("# Threads\n")
    (bank / "learn.zh.md").write_text("# 线程\n")
    errors, pkg = build.build(root, "abc1234")
    assert errors == []
    assert {k: pkg[k] for k in ("schemaVersion", "slug", "kind", "sourceCommit", "runtime", "gpus")} == {
        "schemaVersion": 5,
        "slug": "learn-demo",
        "kind": "learn",
        "sourceCommit": "abc1234",
        "runtime": "cuda@1",
        "gpus": ["t4"],
    }
    assert pkg["parts"][0]["chapters"][0]["stages"] == [
        {
            "slug": "map",
            "title": {"en": "Map", "zh": "映射"},
            "status": "published",
            "summary": {"en": "One thread per element.", "zh": "一个线程算一个元素。"},
        },
        {"slug": "later", "title": {"en": "Later", "zh": "以后"}, "status": "wip"},
    ]
    assert list(pkg["stages"]) == ["map"]
    stage = pkg["stages"]["map"]
    assert set(stage) == {
        "type",
        "difficulty",
        "accessTier",
        "gpus",
        "statement",
        "learn",
        "starterCode",
        "problemPy",
    }
    assert stage["type"] == "code"
    assert stage["difficulty"] == "easy"
    assert stage["gpus"] == ["t4"]
    assert stage["statement"]["en"] == "# Map\n"
    assert stage["learn"] == {"en": "# Threads\n", "zh": "# 线程\n"}
    assert stage["starterCode"] == [
        {
            "language": "cuda",
            "fileName": "solution.cu",
            "fileContent": (bank / "starter/cuda/solution.cu").read_text(),
        }
    ]
    assert stage["problemPy"] == (bank / "problem.py").read_text()


def test_package_carries_the_harness(bank):
    (bank / "harness.cu").write_text('#include "solution.cu"\n')
    _, pkg = build.build(course(bank), "abc1234")
    assert pkg["stages"]["map"]["harnessCu"] == '#include "solution.cu"\n'


def test_quiz_package(quiz):
    root = course(quiz)
    (quiz / "learn.zh.md").write_text("# 线程\n")
    errors, pkg = build.build(root, "abc1234")
    assert errors == []
    stage = pkg["stages"]["check"]
    assert {k: stage[k] for k in ("type", "accessTier", "learn")} == {
        "type": "quiz",
        "accessTier": "free",
        "learn": {"en": "TODO\n", "zh": "# 线程\n"},
    }
    threads, index, size, qualifier = stage["questions"]
    assert threads == {
        "id": "threads",
        "type": "single",
        "answer": 2,
        "text": {
            "en": {
                "prompt": "How many threads does `kernel<<<2, 4>>>()` launch?",
                "options": ["2", "4", "8", "6"],
                "explanation": "Two blocks of four threads.",
                "feedback": {"0": "That is the number of blocks."},
            },
            "zh": {
                "prompt": "`kernel<<<2, 4>>>()` 启动多少个线程？",
                "options": ["2", "4", "8", "6"],
                "explanation": "两个块，每块四个线程。",
                "feedback": {"0": "这是块数。"},
            },
        },
    }
    assert index["answers"] == [0, 2]
    assert set(index["text"]["en"]) == {"prompt", "options", "explanation"}
    assert size["accept"] == {"numeric": 4, "tolerance": 0}
    assert set(size["text"]["zh"]) == {"prompt", "explanation"}
    assert qualifier["accept"] == {"exact": ["__global__"], "caseSensitive": False}


def test_draft_has_no_content(bank):
    _, pkg = build.build(course(bank, "draft"), "abc1234")
    assert pkg["stages"] == {}
    assert pkg["parts"][0]["chapters"][0]["stages"][0]["status"] == "draft"


def test_refuses_errors(bank):
    root = bank.parent.parent
    (root / "course.yml").write_text((root / "course.yml").read_text().replace("learn-demo", "Learn_Demo"))
    (bank / "problem.zh.md").unlink()
    errors, pkg = build.build(root, "abc1234")
    assert pkg is None
    assert "course.yml: slug must be lowercase letters, digits and single '-'" in errors
    assert "course.yml: summary must have en and zh, strings of 1-500 characters" in errors
    assert build.GPUS_ERROR in errors
    assert "map/problem.zh.md: missing" in errors
