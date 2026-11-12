"""Build the course package bootgpu-api serves (its docs/00-design.md §7): one JSON file with the catalog
and every published stage.

Usage: uv run scripts/build.py [-o build/<slug>.json]

Refuses on uncommitted changes (sourceCommit must name what was built) and on any validate.py error.
Only the files named below go in.
"""

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import validate
from validate import ROOT, SLUG, as_text, course_stages, feedback_index, read_yaml, title_errors

SCHEMA_VERSION = 5
GPUS_ERROR = "bootgpu.toml: gpus must be a non-empty list of distinct GPU names"


def header_errors(course: Any) -> list[str]:
    if not isinstance(course, dict):
        return ["course.yml: must be a mapping"]
    errors = (
        []
        if isinstance(course.get("slug"), str) and SLUG.match(course["slug"])
        else ["course.yml: slug must be lowercase letters, digits and single '-'"]
    )
    errors += title_errors("course.yml", course.get("title"))
    summary = course.get("summary")
    if not (
        isinstance(summary, dict)
        and set(summary) == {"en", "zh"}
        and all(isinstance(s, str) and 1 <= len(s.strip()) <= 500 for s in summary.values())
    ):
        errors.append("course.yml: summary must have en and zh, strings of 1-500 characters")
    return errors


def course_gpus(toml: dict) -> list[str]:
    gpus = toml.get("gpus")
    ok = isinstance(gpus, list) and gpus and all(isinstance(g, str) for g in gpus)
    return gpus if ok and len(set(gpus)) == len(gpus) else []


def stage_package(problem_dir: Path, gpus: list[str]) -> dict:
    if (problem_dir / validate.QUIZ).is_file():
        return quiz_package(problem_dir)
    meta, _ = read_yaml(problem_dir / "problem.yml")
    harness = problem_dir / validate.HARNESS
    return {
        "type": "code",
        "difficulty": meta["difficulty"],
        "accessTier": "free",
        "gpus": gpus,
        "statement": {
            "en": (problem_dir / "problem.md").read_text(),
            "zh": (problem_dir / "problem.zh.md").read_text(),
        },
        "learn": {
            "en": (problem_dir / "learn.md").read_text(),
            "zh": (problem_dir / "learn.zh.md").read_text(),
        },
        "starterCode": [
            {
                "language": "cuda",
                "fileName": "solution.cu",
                "fileContent": (problem_dir / "starter/cuda/solution.cu").read_text(),
            }
        ],
        "problemPy": (problem_dir / "problem.py").read_text(),
    } | ({"harnessCu": harness.read_text()} if harness.is_file() else {})


def text_package(q: dict) -> dict:
    return (
        {"prompt": q["prompt"]}
        | ({"options": [as_text(o) for o in q["options"]]} if "options" in q else {})
        | {"explanation": q["explanation"]}
        | (
            {"feedback": {str(feedback_index(k)): v for k, v in q["feedback"].items()}}
            if "feedback" in q
            else {}
        )
    )


def answer_package(q: dict) -> dict:
    """The answer with the optional fields filled in, as bootgpu-api grades it."""
    if q["type"] == "single":
        return {"answer": q["answer"]}
    if q["type"] == "multiple":
        return {"answers": sorted(q["answers"])}
    a = q["accept"]
    if "exact" in a:
        return {
            "accept": {
                "exact": [as_text(x) for x in a["exact"]],
                "caseSensitive": a.get("caseSensitive", False),
            }
        }
    if "numeric" in a:
        return {"accept": {"numeric": a["numeric"], "tolerance": a.get("tolerance", 0)}}
    return {"accept": {"sequence": a["sequence"]}}


def quiz_package(problem_dir: Path) -> dict:
    """bootgpu-api docs/02-quiz.md §3; validate.check_quiz has passed."""
    en, _ = read_yaml(problem_dir / validate.QUIZ)
    zh, _ = read_yaml(problem_dir / validate.QUIZ_ZH)
    return {
        "type": "quiz",
        "accessTier": "free",
        "learn": {
            "en": (problem_dir / "learn.md").read_text(),
            "zh": (problem_dir / "learn.zh.md").read_text(),
        },
        "questions": [
            {"id": q["id"], "type": q["type"]}
            | answer_package(q)
            | {"text": {"en": text_package(q), "zh": text_package(t)}}
            for q, t in zip(en["questions"], zh["questions"], strict=True)
        ],
    }


def build(root: Path, commit: str) -> tuple[list[str], dict | None]:
    """(errors, the package); the package is None when there are errors."""
    course, err = read_yaml(root / "course.yml")
    errors = [f"course.yml: {err}"] if err else header_errors(course)
    errors += validate.validate_course(root)
    try:
        toml = tomllib.loads((root / "bootgpu.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        toml = {}
    gpus = course_gpus(toml)
    if not gpus:
        errors.append(GPUS_ERROR)
    pdir = root / "problems"
    for d in sorted(p for p in pdir.iterdir() if p.is_dir()) if pdir.is_dir() else []:
        errors += [f"{d.name}/{e}" for e in validate.validate(d)]
    if errors:
        return errors, None
    _, stages = course_stages(root)
    parts = []
    for part in course["parts"]:
        chapters = []
        for c in part["chapters"]:
            data, _ = read_yaml(root / "chapters" / c / "chapter.yml")
            chapters.append(
                {
                    "slug": c,
                    "title": data["title"],
                    "stages": [
                        {"slug": s["slug"], "title": s["title"], "status": s["status"]}
                        for s in data["stages"]
                    ],
                }
            )
        parts.append({"title": part["title"], "chapters": chapters})
    return [], {
        "schemaVersion": SCHEMA_VERSION,
        "slug": course["slug"],
        "kind": toml["kind"],
        "sourceCommit": commit,
        "title": course["title"],
        "summary": course["summary"],
        "runtime": toml["runtime"],
        "gpus": gpus,
        "parts": parts,
        "stages": {
            slug: stage_package(pdir / slug, gpus) for slug, s in stages.items() if s["status"] == "published"
        },
    }


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-o", "--output", type=Path, help="default: build/<slug>.json")
    args = ap.parse_args(argv)
    if git("status", "--porcelain", "--untracked-files=all"):
        print("error: uncommitted changes; commit or stash them first", file=sys.stderr)
        return 1
    errors, pkg = build(ROOT, git("rev-parse", "HEAD").strip())
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    if pkg is None:
        return 1
    out = args.output or ROOT / "build" / f"{pkg['slug']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pkg, ensure_ascii=False, indent=2) + "\n")
    print(f"{out}: {len(pkg['stages'])} published stages, {out.stat().st_size / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
