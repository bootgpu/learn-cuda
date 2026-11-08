"""Check the course without a GPU: course.yml and the chapters, then each problem's layout, problem.yml,
imports and seeds, and the test cases.

Usage: uv run scripts/validate.py                    the course and every problem
       uv run scripts/validate.py problems/<slug> …  only those problems

scripts/verify.py runs the problem checks first, then grades the solutions from course-solutions on the GPU.
"""

import ast
import ctypes
import importlib.util
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
# problem.py imports bootgpu.problem from this repository's copy.
sys.path.insert(0, str(ROOT))

SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
FILES = (
    "problem.yml",
    "problem.md",
    "problem.zh.md",
    "learn.md",
    "learn.zh.md",
    "problem.py",
    "starter/cuda/solution.cu",
)
# Passing code; it lives in the private bootgpu/course-solutions.
PRIVATE = ("solution", "variants")
DIFFICULTIES = ("easy", "medium", "hard")
STATUSES = ("wip", "draft", "published")
ALLOWED_MODULES = {"bootgpu.problem"}
SEED_CALLS = {"manual_seed", "manual_seed_all", "seed", "set_rng_state"}
ROLES = ("in", "out", "inout")
# The course series these checks are for; bootgpu-api turns community features on or off by it.
KIND = "learn"
# Per tensor in every case; a stage that needs more sets max_tensor_elements in problem.yml.
MAX_TENSOR_ELEMENTS = 10_000


def read_yaml(path: Path) -> tuple[Any, str | None]:
    try:
        return yaml.safe_load(path.read_text()), None
    except OSError:
        return None, "missing"
    except yaml.YAMLError as e:
        return None, f"not valid YAML: {str(e).splitlines()[0]}"


def title_errors(where: str, title: Any) -> list[str]:
    if not isinstance(title, dict):
        return [f"{where}: title must be a mapping with en and zh"]
    errors = [f"{where}: unknown key title.{k}" for k in title if k not in ("en", "zh")]
    for lang in ("en", "zh"):
        t = title.get(lang)
        if not (isinstance(t, str) and 1 <= len(t.strip()) <= 200):
            errors.append(f"{where}: title.{lang} must be a string of 1-200 characters")
    return errors


def kind_errors(root: Path) -> list[str]:
    try:
        kind = tomllib.loads((root / "bootgpu.toml").read_text()).get("kind")
    except (OSError, tomllib.TOMLDecodeError):
        kind = None
    errors = [] if kind == KIND else [f'bootgpu.toml: kind must be "{KIND}"']
    course, _ = read_yaml(root / "course.yml")
    slug = course.get("slug") if isinstance(course, dict) else None
    if not (isinstance(slug, str) and slug.startswith(f"{KIND}-")):
        errors.append(f"course.yml: slug must start with {KIND}-")
    return errors


def check_chapter(path: Path) -> tuple[list[str], list[dict]]:
    """(errors, the well-formed stages)."""
    where = f"chapters/{path.parent.name}/chapter.yml"
    data, err = read_yaml(path)
    if err:
        return [f"{where}: {err}"], []
    if not isinstance(data, dict):
        return [f"{where}: must be a mapping with title and stages"], []
    errors = [f"{where}: unknown key {k!r}" for k in data if k not in ("title", "stages")]
    errors += title_errors(where, data.get("title"))
    stages = data.get("stages")
    if not (isinstance(stages, list) and stages):
        return [*errors, f"{where}: stages must be a non-empty list"], []
    good = []
    for i, s in enumerate(stages, 1):
        w = f"{where}: stages[{i}]"
        if not isinstance(s, dict):
            errors.append(f"{w} must be a mapping with slug, title and status")
            continue
        e = [f"{w}: unknown key {k!r}" for k in s if k not in ("slug", "title", "status")]
        if not (isinstance(s.get("slug"), str) and SLUG.match(s["slug"])):
            e.append(f"{w}: slug must be lowercase letters, digits and single '-'")
        e += title_errors(w, s.get("title"))
        if s.get("status") not in STATUSES:
            e.append(f"{w}: status must be one of {', '.join(STATUSES)}")
        errors += e
        if not e:
            good.append(s)
    return errors, good


def course_stages(root: Path) -> tuple[list[str], dict[str, dict]]:
    """(errors, {slug: stage plus its "chapter"}) from course.yml and the chapter.yml files it lists."""
    course, err = read_yaml(root / "course.yml")
    if err:
        return [f"course.yml: {err}"], {}
    parts = course.get("parts") if isinstance(course, dict) else None
    if not (isinstance(parts, list) and parts):
        return ["course.yml: parts must be a non-empty list"], {}
    errors, listed = [], []
    for i, part in enumerate(parts, 1):
        where = f"course.yml: parts[{i}]"
        if not isinstance(part, dict):
            errors.append(f"{where} must be a mapping with title and chapters")
            continue
        errors += title_errors(where, part.get("title"))
        chapters = part.get("chapters")
        if not (isinstance(chapters, list) and chapters and all(isinstance(c, str) for c in chapters)):
            errors.append(f"{where}: chapters must be a non-empty list of chapter slugs")
            continue
        listed += chapters
    errors += [
        f"course.yml: chapter {c!r} is listed twice"
        for c in sorted({c for c in listed if listed.count(c) > 1})
    ]
    cdir = root / "chapters"
    dirs = {p.name for p in cdir.iterdir() if p.is_dir()} if cdir.is_dir() else set()
    errors += [f"chapters/{c}: not listed in course.yml" for c in sorted(dirs - set(listed))]
    stages: dict[str, dict] = {}
    for c in dict.fromkeys(listed):
        e, good = check_chapter(cdir / c / "chapter.yml")
        errors += e
        for s in good:
            if s["slug"] in stages:
                errors.append(
                    f"stage {s['slug']!r} is in chapters/{stages[s['slug']]['chapter']} and chapters/{c}"
                )
            else:
                stages[s["slug"]] = {**s, "chapter": c}
    return errors, stages


def validate_course(root: Path) -> list[str]:
    errors, stages = course_stages(root)
    errors += kind_errors(root)
    for slug, s in stages.items():
        if s["status"] != "wip" and not (root / "problems" / slug).is_dir():
            errors.append(
                f"chapters/{s['chapter']}: stage {slug} is {s['status']} but problems/{slug} is missing"
            )
    return errors


def check_yml(path: Path, stage_title: Any) -> list[str]:
    meta, err = read_yaml(path)
    if err:
        return [f"problem.yml: {err}"]
    if not isinstance(meta, dict):
        return ["problem.yml: must be a mapping with title and difficulty"]
    keys = ("title", "difficulty", "max_tensor_elements")
    errors = [f"problem.yml: unknown key {k!r}" for k in meta if k not in keys]
    errors += title_errors("problem.yml", meta.get("title"))
    if stage_title is not None and meta.get("title") != stage_title:
        errors.append("problem.yml: title differs from the stage's title in chapter.yml")
    if meta.get("difficulty") not in DIFFICULTIES:
        errors.append(f"problem.yml: difficulty must be one of {', '.join(DIFFICULTIES)}")
    if "max_tensor_elements" in meta and tensor_limit(meta) is None:
        errors.append(f"problem.yml: max_tensor_elements must be an integer above {MAX_TENSOR_ELEMENTS}")
    return errors


def tensor_limit(meta: Any) -> int | None:
    """None when problem.yml sets an invalid max_tensor_elements."""
    if not isinstance(meta, dict) or "max_tensor_elements" not in meta:
        return MAX_TENSOR_ELEMENTS
    n = meta["max_tensor_elements"]
    return n if type(n) is int and n > MAX_TENSOR_ELEMENTS else None


def check_source(tree: ast.AST) -> list[str]:
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = ["." * node.level + (node.module or "")]
        else:
            modules = []
        for m in modules:
            top = m.split(".")[0]
            if not (
                m in ALLOWED_MODULES
                or top == "torch"
                or (top in sys.stdlib_module_names and top != "bootgpu")
            ):
                errors.append(
                    f"problem.py:{node.lineno}: import of {m!r} is not allowed"
                    " (only torch, the standard library and bootgpu.problem)"
                )
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
            if name in SEED_CALLS:
                errors.append(f"problem.py:{node.lineno}: {name}() is not allowed; bg-run seeds every run")
    return errors


def load_problem(path: Path, slug: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"problem_{slug.replace('-', '_')}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Problem(device="cpu")


def generate_cases(pb: Any, seed: int) -> list[tuple[str, dict[str, Any]]]:
    torch.manual_seed(seed)
    examples = pb.examples()
    if not isinstance(examples, list) or not examples:
        raise ValueError("examples() must return a non-empty list")
    tests = pb.tests()
    names = [f"example_{i}" for i in range(1, len(examples) + 1)]
    names += [f"test_{i}" for i in range(1, len(tests) + 1)]
    return list(zip(names, [*examples, *tests], strict=True))


def signature_errors(sig: Any, args: dict[str, Any]) -> list[str]:
    if not isinstance(sig, dict) or not sig:
        return ["signature() must return a non-empty dict"]
    if set(sig) != set(args):
        return [f"test case args {sorted(args)} do not match the signature {list(sig)}"]
    for k, entry in sig.items():
        if not (isinstance(entry, tuple) and len(entry) == 2):
            return [f"{k}: expected (ctypes type, role), got {entry!r}"]
        ctype, role = entry
        if role not in ROLES:
            return [f"{k}: role must be one of {ROLES}, got {role!r}"]
        if not (isinstance(ctype, type) and issubclass(ctype, ctypes._SimpleCData)):
            return [f"{k}: {ctype!r} is not a ctypes scalar type"]
        if torch.is_tensor(args[k]):
            if ctype is not ctypes.c_void_p:
                return [f"{k}: tensors must be passed as ctypes.c_void_p"]
        elif ctype is ctypes.c_void_p or role != "in":
            return [f"{k}: scalars must be role 'in' with a non-pointer ctypes type"]
    return []


def same_case(a: dict[str, Any], b: dict[str, Any], skip: set[str]) -> bool:
    """skip: `out` buffers, usually torch.empty; the judge overwrites them with sentinels."""
    if a.keys() != b.keys():
        return False
    return all(
        torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) and torch.is_tensor(b[k]) else a[k] == b[k]
        for k in a
        if k not in skip
    )


def check_problem(path: Path, slug: str, limit: int = MAX_TENSOR_ELEMENTS) -> list[str]:
    try:
        pb = load_problem(path, slug)
        sig = pb.signature()
        cases = generate_cases(pb, 0)
        again = generate_cases(pb, 1)
    except Exception as e:
        return [f"problem.py: {type(e).__name__}: {e}"]
    errors = []
    if hasattr(pb, "performance"):
        errors.append("problem.py: performance() is not allowed; learn courses only check correctness")
    for name, args in cases:
        errors += [f"problem.py: {name}: {e}" for e in signature_errors(sig, args)]
    for name, args in [*cases, *again]:
        for k, v in args.items():
            if torch.is_tensor(v) and v.numel() > limit:
                errors.append(
                    f"problem.py: {name}: {k} has {v.numel():,} elements, above {limit:,};"
                    " learn tests stay small (max_tensor_elements in problem.yml raises it for one stage)"
                )
    errors = list(dict.fromkeys(errors))
    items = sig.items() if isinstance(sig, dict) else []
    outs = {k for k, e in items if isinstance(e, tuple) and e[1:] == ("out",)}
    for (name, a), (_, b) in zip(cases, again, strict=True):
        if name.startswith("example_") and not same_case(a, b, outs):
            errors.append(f"problem.py: {name} changes with the seed; examples must use fixed values")
    return errors


def validate(problem_dir: Path) -> list[str]:
    root, slug = problem_dir.parent.parent, problem_dir.name
    errors = [] if SLUG.match(slug) else ["slug must be lowercase letters, digits and single '-'"]
    errors += [f"{f}: missing" for f in FILES if not (problem_dir / f).is_file()]
    errors += [f"{d}/: belongs in bootgpu/course-solutions" for d in PRIVATE if (problem_dir / d).exists()]
    stage = course_stages(root)[1].get(slug)
    if stage is None:
        errors.append("not a stage in any chapter.yml listed by course.yml")
    elif stage["status"] == "wip":
        errors.append(f"the stage is wip in chapters/{stage['chapter']}/chapter.yml; use draft or published")
    limit = MAX_TENSOR_ELEMENTS
    if (problem_dir / "problem.yml").is_file():
        errors += check_yml(problem_dir / "problem.yml", stage and stage["title"])
        limit = tensor_limit(read_yaml(problem_dir / "problem.yml")[0]) or MAX_TENSOR_ELEMENTS
    path = problem_dir / "problem.py"
    if path.is_file():
        try:
            source_errors = check_source(ast.parse(path.read_text(), "problem.py"))
        except SyntaxError as e:
            source_errors = [f"problem.py:{e.lineno}: {e.msg}"]
        errors += source_errors
        if not source_errors:
            errors += check_problem(path, slug, limit)
    return errors


def main(argv: list[str]) -> int:
    course_errors = [] if argv else validate_course(ROOT)
    for e in course_errors:
        print(f"error: {e}", file=sys.stderr)
    pdir = ROOT / "problems"
    dirs = [Path(a) for a in argv] or (
        sorted(p for p in pdir.iterdir() if p.is_dir()) if pdir.is_dir() else []
    )
    failed = 0
    for d in dirs:
        errors = validate(d.resolve())
        failed += bool(errors)
        for e in errors:
            print(f"error: {d.name}/{e}", file=sys.stderr)
    print(
        f"{len(dirs) - failed}/{len(dirs)} problems valid"
        + ("; the course has errors" if course_errors else "")
    )
    return 1 if failed or course_errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
