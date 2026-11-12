"""Check the course without a GPU: course.yml and the chapters, then each problem's layout, problem.yml,
imports and seeds, and the test cases. A quiz stage (quiz.yml, bootgpu-api docs/02-quiz.md §2) gets the
quiz checks instead; its question-writing warnings are printed but do not fail.

Usage: uv run scripts/validate.py                    the course and every problem
       uv run scripts/validate.py problems/<slug> …  only those problems

scripts/verify.py runs the problem checks first, then grades the solutions from course-solutions on the GPU.
"""

import ast
import ctypes
import importlib.util
import math
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
# Optional: a stage that fixes solve() ships it here; the judge compiles it in place of solution.cu.
HARNESS = "harness.cu"
HARNESS_FIRST_LINE = '#include "solution.cu"'
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

# A quiz stage: quiz.yml holds the questions, answers and English, quiz.zh.yml the Chinese text.
QUIZ = "quiz.yml"
QUIZ_ZH = "quiz.zh.yml"
QUIZ_FILES = (QUIZ, QUIZ_ZH, "learn.md", "learn.zh.md")
CODE_FILES = ("problem.yml", "problem.md", "problem.zh.md", "problem.py", "starter", HARNESS)
QUESTION_TYPES = {"single": ("options", "answer"), "multiple": ("options", "answers"), "blank": ("accept",)}
QUESTION_KEYS = ("id", "type", "prompt", "options", "answer", "answers", "accept", "explanation", "feedback")
TEXT_KEYS = ("id", "prompt", "options", "explanation", "feedback")
QUESTIONS_MAX = 10
QUESTION_ID_MAX = 40
OPTIONS_MIN, OPTIONS_MAX = 2, 6
MARKDOWN_MAX_BYTES = 4096
CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)
RAW_HTML = re.compile(r"<!--|</?[A-Za-z][A-Za-z0-9-]*(\s[^<>]*)?/?>")
# Options that make a question guessable or vague (whole option / anywhere in it).
CATCH_ALL = {"都对", "都不对", "都是", "都不是"}
CATCH_ALL_PHRASES = (
    "all of the above",
    "none of the above",
    "both of the above",
    "以上都",
    "以上皆",
    "以上均",
)
# The correct option may not be this much longer than every distractor.
LENGTH_RATIO = 1.5


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


def as_text(v: Any) -> Any:
    """YAML reads `- 10` as a number; options and accepted answers are text either way."""
    return str(v) if type(v) in (int, float) else v


def is_number(v: Any) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def feedback_index(key: Any) -> int | None:
    """YAML reads `0:` as an int key and `"0":` as a string."""
    if type(key) is int:
        return key
    return int(key) if isinstance(key, str) and key.isascii() and key.isdigit() else None


def markdown_errors(where: str, text: Any) -> list[str]:
    if not (isinstance(text, str) and text.strip()):
        return [f"{where} must be non-empty text"]
    errors = []
    if len(text.encode()) > MARKDOWN_MAX_BYTES:
        errors.append(f"{where} must be at most {MARKDOWN_MAX_BYTES} bytes")
    if RAW_HTML.search(CODE.sub("", text)):
        errors.append(f"{where}: raw HTML is not allowed (put code in backticks)")
    return errors


def option_count(q: dict) -> int:
    """0 when the options are missing or malformed."""
    options = q.get("options")
    ok = isinstance(options, list) and OPTIONS_MIN <= len(options) <= OPTIONS_MAX
    return len(options) if ok else 0


def text_errors(where: str, q: dict) -> list[str]:
    """prompt, options, explanation and feedback: what quiz.yml and quiz.zh.yml both write."""
    errors = markdown_errors(f"{where}: prompt", q.get("prompt"))
    errors += markdown_errors(f"{where}: explanation", q.get("explanation"))
    if "options" in q:
        if not option_count(q):
            errors.append(f"{where}: options must be a list of {OPTIONS_MIN}-{OPTIONS_MAX} options")
        else:
            texts = [as_text(o) for o in q["options"]]
            for i, o in enumerate(texts):
                errors += markdown_errors(f"{where}: options[{i}]", o)
            seen = [o.strip() for o in texts if isinstance(o, str)]
            errors += [
                f"{where}: option {o!r} appears twice" for o in sorted({o for o in seen if seen.count(o) > 1})
            ]
    if "feedback" in q:
        fb = q["feedback"]
        if not (isinstance(fb, dict) and fb and all(feedback_index(k) is not None for k in fb)):
            errors.append(f"{where}: feedback must map option indexes to text")
        else:
            for k, v in fb.items():
                errors += markdown_errors(f"{where}: feedback.{k}", v)
    return errors


def accept_errors(where: str, a: Any) -> list[str]:
    exact = isinstance(a, dict) and a.keys() <= {"exact", "caseSensitive"} and "exact" in a
    numeric = isinstance(a, dict) and a.keys() <= {"numeric", "tolerance"} and "numeric" in a
    sequence = isinstance(a, dict) and a.keys() == {"sequence"}
    ok = (
        (
            exact
            and isinstance(a["exact"], list)
            and len(a["exact"]) > 0
            and all(isinstance(as_text(x), str) and as_text(x).strip() for x in a["exact"])
            and type(a.get("caseSensitive", False)) is bool
        )
        or (
            numeric
            and is_number(a["numeric"])
            and is_number(a.get("tolerance", 0))
            and a.get("tolerance", 0) >= 0
        )
        or (
            sequence
            and isinstance(a["sequence"], list)
            and len(a["sequence"]) > 0
            and all(is_number(x) or (isinstance(x, str) and x.strip()) for x in a["sequence"])
        )
    )
    if ok:
        return []
    return [
        f"{where}: accept must be {{exact: [text, ...], caseSensitive?: bool}},"
        " {numeric: number, tolerance?: number >= 0} or {sequence: [number or text, ...]}"
    ]


def question_errors(where: str, q: Any) -> list[str]:
    if not isinstance(q, dict):
        return [f"{where} must be a mapping"]
    errors = [f"{where}: unknown key {k!r}" for k in q if k not in QUESTION_KEYS]
    qid = q.get("id")
    if not (isinstance(qid, str) and SLUG.match(qid) and len(qid) <= QUESTION_ID_MAX):
        errors.append(
            f"{where}: id must be lowercase letters, digits and single '-',"
            f" at most {QUESTION_ID_MAX} characters"
        )
    qtype = q.get("type")
    if qtype not in QUESTION_TYPES:
        return [*errors, f"{where}: type must be one of {', '.join(QUESTION_TYPES)}"]
    for k in ("options", "answer", "answers", "accept"):
        if k in QUESTION_TYPES[qtype] and k not in q:
            errors.append(f"{where}: a {qtype} question needs {k}")
        if k not in QUESTION_TYPES[qtype] and k in q:
            errors.append(f"{where}: a {qtype} question cannot have {k}")
    if "feedback" in q and qtype != "single":
        errors.append(f"{where}: feedback is only for single questions")
    errors += text_errors(where, q)
    n = option_count(q)

    def is_index(i: Any) -> bool:
        return type(i) is int and 0 <= i < n

    if n and qtype == "single" and "answer" in q and not is_index(q["answer"]):
        errors.append(f"{where}: answer must be an option index (0-{n - 1})")
    if n and qtype == "multiple" and "answers" in q:
        a = q["answers"]
        if not (isinstance(a, list) and a and all(is_index(i) for i in a)):
            errors.append(f"{where}: answers must be a non-empty list of option indexes (0-{n - 1})")
        elif len(set(a)) < len(a):
            errors.append(f"{where}: answers lists an option twice")
        elif len(a) == n:
            errors.append(f"{where}: answers cannot be every option")
    if n and qtype == "single" and isinstance(q.get("feedback"), dict):
        for k in q["feedback"]:
            i = feedback_index(k)
            if i is not None and not is_index(i):
                errors.append(f"{where}: feedback.{k} is not an option index (0-{n - 1})")
            elif i is not None and i == q.get("answer"):
                errors.append(f"{where}: feedback.{k} is the correct option (feedback is for wrong options)")
    if qtype == "blank" and "accept" in q:
        errors += accept_errors(where, q["accept"])
    return errors


def questions_of(path: Path) -> tuple[list[str], list]:
    data, err = read_yaml(path)
    if err:
        return [f"{path.name}: {err}"], []
    if not (isinstance(data, dict) and data.keys() == {"questions"} and isinstance(data["questions"], list)):
        return [f"{path.name}: must be a mapping with only questions, a list"], []
    return [], data["questions"]


def plain(markdown: str) -> str:
    """Rough plain text for length comparisons: link text kept, emphasis and code marks dropped."""
    return re.sub(r"[`*_~]", "", re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", markdown)).strip()


def lint_quiz(questions: list[dict], texts: dict[str, list[dict]]) -> tuple[list[str], list[str]]:
    """The mechanical question-writing rules, on each language's options: (errors, warnings)."""
    errors, warnings = [], []
    for i, q in enumerate(questions):
        for file, qs in texts.items():
            if "options" not in qs[i]:
                continue
            where = f"{file}: {q['id']}"
            options = [plain(as_text(o)) for o in qs[i]["options"]]
            for j, o in enumerate(options):
                if o.lower() in CATCH_ALL or any(p in o.lower() for p in CATCH_ALL_PHRASES):
                    errors.append(f"{where}: option {j} {o!r} is a catch-all option; write a real answer")
            if q["type"] != "single":
                continue
            correct = len(options[q["answer"]])
            longest = max(len(o) for j, o in enumerate(options) if j != q["answer"])
            if correct > longest * LENGTH_RATIO:
                errors.append(
                    f"{where}: the correct option ({correct} characters) is more than {LENGTH_RATIO}x every"
                    f" other option (longest {longest}); move the reasoning into the explanation"
                )
            elif correct > longest:
                warnings.append(
                    f"{where}: the correct option is the longest one ({correct} vs {longest} characters)"
                )
    positions: dict[int, list[str]] = {}
    for q in questions:
        if q["type"] == "single":
            positions.setdefault(q["answer"], []).append(q["id"])
    for position, ids in sorted(positions.items()):
        if len(ids) >= 3:
            warnings.append(
                f"{QUIZ}: {len(ids)} single questions have the answer at option {position}"
                f" ({', '.join(ids)}); vary the position"
            )
    return errors, warnings


def check_quiz(problem_dir: Path) -> tuple[list[str], list[str]]:
    """(errors, warnings) for quiz.yml and quiz.zh.yml; the lint runs once the structure is right."""
    errors, questions = questions_of(problem_dir / QUIZ)
    if errors:
        return errors, []
    if not 1 <= len(questions) <= QUESTIONS_MAX:
        return [f"{QUIZ}: questions must list 1-{QUESTIONS_MAX} questions"], []
    for i, q in enumerate(questions):
        errors += question_errors(f"{QUIZ}: questions[{i}]", q)
    if errors:
        return errors, []
    ids = [q["id"] for q in questions]
    errors += [f"{QUIZ}: question id {d} appears twice" for d in sorted({d for d in ids if ids.count(d) > 1})]
    e, zh = questions_of(problem_dir / QUIZ_ZH)
    errors += e
    if errors:
        return errors, []
    if [t.get("id") if isinstance(t, dict) else None for t in zh] != ids:
        return [f"{QUIZ_ZH}: questions must have the same ids in the same order as {QUIZ}"], []
    for q, t in zip(questions, zh, strict=True):
        where = f"{QUIZ_ZH}: {q['id']}"
        errors += [f"{where}: unknown key {k!r}" for k in t if k not in TEXT_KEYS]
        errors += text_errors(where, t)
        if ("options" in q) != ("options" in t):
            errors.append(f"{where}: options {'are missing' if 'options' in q else 'are not expected'}")
        elif "options" in q and option_count(t) and len(t["options"]) != len(q["options"]):
            errors.append(f"{where}: {len(t['options'])} options, {QUIZ} has {len(q['options'])}")
        fb_en = {feedback_index(k) for k in q.get("feedback") or {}}
        fb_zh = {feedback_index(k) for k in t["feedback"]} if isinstance(t.get("feedback"), dict) else set()
        if fb_en != fb_zh:
            errors.append(f"{where}: feedback must cover the same options as {QUIZ}")
    if errors:
        return errors, []
    return lint_quiz(questions, {QUIZ: questions, QUIZ_ZH: zh})


def validate(problem_dir: Path, warnings: list[str] | None = None) -> list[str]:
    """The problem's errors; a quiz stage's question-writing warnings go to `warnings` when given."""
    root, slug = problem_dir.parent.parent, problem_dir.name
    quiz = (problem_dir / QUIZ).is_file()
    errors = [] if SLUG.match(slug) else ["slug must be lowercase letters, digits and single '-'"]
    errors += [f"{f}: missing" for f in (QUIZ_FILES if quiz else FILES) if not (problem_dir / f).is_file()]
    if quiz:
        errors += [
            f"{f}: a quiz stage (it has {QUIZ}) has no {f}" for f in CODE_FILES if (problem_dir / f).exists()
        ]
    errors += [f"{d}/: belongs in bootgpu/course-solutions" for d in PRIVATE if (problem_dir / d).exists()]
    harness = problem_dir / HARNESS
    if harness.is_file() and harness.read_text().split("\n", 1)[0].strip() != HARNESS_FIRST_LINE:
        errors.append(f"{HARNESS}: the first line must be {HARNESS_FIRST_LINE}")
    stage = course_stages(root)[1].get(slug)
    if stage is None:
        errors.append("not a stage in any chapter.yml listed by course.yml")
    elif stage["status"] == "wip":
        errors.append(f"the stage is wip in chapters/{stage['chapter']}/chapter.yml; use draft or published")
    if quiz:
        if (problem_dir / QUIZ_ZH).is_file():
            e, w = check_quiz(problem_dir)
            errors += e
            if warnings is not None:
                warnings += w
        return errors
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
        warnings: list[str] = []
        errors = validate(d.resolve(), warnings)
        failed += bool(errors)
        for e in errors:
            print(f"error: {d.name}/{e}", file=sys.stderr)
        for w in warnings:
            print(f"warning: {d.name}/{w}", file=sys.stderr)
    print(
        f"{len(dirs) - failed}/{len(dirs)} problems valid"
        + ("; the course has errors" if course_errors else "")
    )
    return 1 if failed or course_errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
