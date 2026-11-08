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
        "schemaVersion": 4,
        "slug": "learn-demo",
        "kind": "learn",
        "sourceCommit": "abc1234",
        "runtime": "cuda@1",
        "gpus": ["t4"],
    }
    assert pkg["parts"][0]["chapters"][0]["stages"] == [
        {"slug": "map", "title": {"en": "Map", "zh": "映射"}, "status": "published"},
        {"slug": "later", "title": {"en": "Later", "zh": "以后"}, "status": "wip"},
    ]
    assert list(pkg["stages"]) == ["map"]
    stage = pkg["stages"]["map"]
    assert set(stage) == {
        "difficulty",
        "accessTier",
        "gpus",
        "statement",
        "learn",
        "starterCode",
        "problemPy",
    }
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
