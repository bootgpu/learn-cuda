"""Grade each problem's variants, official solution and starter on the live GPU entry.

Usage: BG_JUDGE_URL=... BG_MODAL_KEY=... BG_MODAL_SECRET=... uv run scripts/verify.py [--solutions DIR]
       [problems/<slug> ...]   (default: all problems)

The official solutions and variants live in the private bootgpu/course-solutions, under <course slug>/
problems/<slug>/; DIR is that course's directory (default: ../course-solutions/<course slug> beside this
repository). validate.py's checks and the solution checks run first. Every file runs RUNS times, each with a
fresh random seed chosen by bg-run.
Each run is a POST /v1/judge (mode submit) read to its result, by the caller rules of sandbox-modal
docs/direct/01 §5. The output never shows variant source or compiler logs, since it may end up in a CI log.
"""

import argparse
import http.client
import json
import os
import re
import secrets
import sys
import time
import tomllib
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import validate

SOLUTION = "solution/cuda/solution.cu"
VARIANTS = "variants/cuda"
RUNS = 3
WORKERS = 4
ATTEMPTS = 3
RETRY_SEC = 10
PASSED = ("finished", None, "PASSED", None, None)
TIMEOUT_SEC = 15  # T of a learn submit in bootgpu-api (its docs/00-design.md §3.1)
QUEUE_SEC = 30  # deadline = send time + this
HEADER_SEC = 31  # no response headers by then: busy
SLACK_SEC = 5  # started, then no result within T + this: the platform lost it
READ_SEC = 15  # a heartbeat comes every 10 s
LOST = "lost track of input"


class Client:
    def __init__(self, url: str, auth: dict[str, str]):
        u = urllib.parse.urlsplit(url)
        self.host, self.auth = u.netloc, auth
        self.conn = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        self.versions: set[str] = set()  # started.version: the sandbox deploy that graded

    def submit(self, body: dict) -> tuple[int, dict]:
        """(200, gradeResponse); (503, busy) without response headers in 31 s or on expired; (0, ...) when the
        platform lost it; else the entry's error answer. A call lost before it ran is sent again once."""
        t0 = time.monotonic()
        sent = time.time()
        data = json.dumps({"mode": "submit", "sentAt": sent, "deadline": sent + QUEUE_SEC} | body).encode()
        for attempt in (1, 2):
            left = HEADER_SEC - (time.monotonic() - t0)
            if left <= 0:
                break
            status, resp = self._once(data, left, body["timeoutSec"])
            if attempt == 2 or not (status == 500 and LOST in str(resp.get("error"))):
                return status, resp
        return 503, {"error": "busy"}

    def _once(self, data: bytes, header_sec: float, timeout_sec: int) -> tuple[int, dict]:
        conn = self.conn(self.host, timeout=header_sec)
        try:
            try:
                conn.request("POST", "/v1/judge", data, {"content-type": "application/json"} | self.auth)
                sock = conn.sock  # conn lets go of it when the response is to close the connection
                r = conn.getresponse()
            except TimeoutError:
                return 503, {"error": "busy"}
            if r.status != 200:
                text = r.read().decode(errors="replace")
                try:
                    return r.status, json.loads(text)
                except ValueError:
                    return r.status, {"error": text[:300]}
            started = None
            while True:
                wait = READ_SEC
                if started is not None:
                    wait = min(wait, started + timeout_sec + SLACK_SEC - time.monotonic())
                if wait <= 0:
                    return 0, {"error": f"no result within T + {SLACK_SEC} s of started"}
                sock.settimeout(wait)
                try:
                    line = r.readline()
                except TimeoutError:
                    if wait < READ_SEC:
                        return 0, {"error": f"no result within T + {SLACK_SEC} s of started"}
                    return 0, {"error": f"no event in {READ_SEC} s"}
                if not line:
                    return 0, {"error": "the stream ended without a result"}
                if not line.strip():
                    continue
                e = json.loads(line)
                if e["type"] == "expired":
                    return 503, {"error": "busy"}
                if e["type"] == "started":
                    started = time.monotonic()
                    self.versions.add(e.get("version") or "unknown")
                if e["type"] == "result":
                    return 200, e["response"]
        finally:
            conn.close()


def outcome(body: dict[str, Any]) -> tuple:
    rr = body.get("runResult") or {}
    failed = [t for t in rr.get("testResults", []) if not t["successful"]]
    first = failed[0] if failed else {}
    return (
        body.get("status"),
        body.get("failureKind"),
        rr.get("status"),
        first.get("name"),
        first.get("kind"),
    )


def matches(expect: tuple | None, got: tuple) -> bool:
    """expect: None for the starter, which may fail tests any way."""
    return got[:3] == ("finished", None, "TESTS_FAILED") if expect is None else got == expect


def grade(
    client: Client, runtime: str, timeout_sec: int, slug: str, name: str, run: int, code: str, problem: str
) -> tuple[tuple, str]:
    """(outcome, note); retries what the platform got wrong, never what the code did."""
    label = re.sub(r"[^A-Za-z0-9_-]", "-", f"verify-{slug}-{name}-{run}")[:46]
    body = {
        "language": "cuda",
        "runtime": runtime,
        "timeoutSec": timeout_sec,
        "code": code,
        "clone": {"files": {"problem.py": problem}},
    }
    note = ""
    for attempt in range(1, ATTEMPTS + 1):
        body["requestId"] = f"{label}-{secrets.token_hex(8)}"  # a retry is a new request
        try:
            status, resp = client.submit(body)
        except (OSError, http.client.HTTPException, ValueError) as e:
            status, resp = 0, {"error": str(e)}
        if status == 200 and resp.get("status") != "system_error":
            rr = resp.get("runResult") or {}
            failed = [t for t in rr.get("testResults", []) if not t["successful"]]
            # A failure message ends with its most specific line, then "seed=N".
            last = failed[0].get("message", "").strip().splitlines()[-2:] if failed else []
            note = " | ".join(last)[:300] if failed else resp.get("detail") or ""
            return outcome(resp), note
        note = f"http {status}: {resp.get('error') or resp.get('detail')}"
        # A 4xx is our request's (or token's) fault; busy comes back as 503.
        if 400 <= status < 500:
            return ("rejected", resp.get("error"), None, None, None), note
        if attempt < ATTEMPTS:
            time.sleep(RETRY_SEC)
    return ("system_error", None, None, None, None), note


def solution_errors(sol_dir: Path) -> list[str]:
    """Every stage has an official solution; variants are optional (a stage that accumulates floats keeps its
    least precise correct solution there)."""
    errors = [] if (sol_dir / SOLUTION).is_file() else [f"{SOLUTION}: missing"]
    vdir = sol_dir / VARIANTS
    if vdir.is_dir():
        errors += [
            f"{VARIANTS}/{f.name}: unexpected file (only correct_*.cu)"
            for f in sorted(vdir.iterdir())
            if not (f.suffix == ".cu" and f.name.startswith("correct_"))
        ]
    return errors


def verify(problem_dir: Path, sol_dir: Path, runtime: str, client: Client) -> bool:
    """sol_dir: the problem's directory in course-solutions."""
    slug = problem_dir.resolve().name
    errors = validate.validate(problem_dir.resolve())
    for e in errors:
        print(f"error: {slug}/{e}", file=sys.stderr)
    sol_errors = solution_errors(sol_dir)
    for e in sol_errors:
        print(f"error: {sol_dir}/{e}", file=sys.stderr)
    if errors or sol_errors:
        return False

    problem = (problem_dir / "problem.py").read_text()
    variants = sorted((sol_dir / VARIANTS).glob("*.cu"))
    files = [(f.stem, f.read_text(), PASSED) for f in variants]
    files.append(("solution", (sol_dir / SOLUTION).read_text(), PASSED))
    files.append(("starter", (problem_dir / "starter/cuda/solution.cu").read_text(), None))
    jobs = [(name, code, expect, run) for name, code, expect in files for run in range(1, RUNS + 1)]
    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(
            pool.map(lambda j: grade(client, runtime, TIMEOUT_SEC, slug, j[0], j[3], j[1], problem), jobs)
        )

    print(f"{slug}: {runtime} graded by sandbox deploy {', '.join(sorted(client.versions)) or 'none'}")
    ok = 0
    for (name, _, expect, run), (got, note) in zip(jobs, results, strict=True):
        good = matches(expect, got)
        ok += good
        line = f"{'ok  ' if good else 'FAIL'}  {slug}/{name} run {run}: {got}"
        if not good:
            line += f"  expected {expect or 'TESTS_FAILED'}" + (f"\n      {note}" if note else "")
        print(line, flush=True)
    print(f"{slug}: {ok}/{len(jobs)} runs as declared")
    return ok == len(jobs)


def client_from_env() -> Client:
    url, key, secret = (os.environ.get(k) for k in ("BG_JUDGE_URL", "BG_MODAL_KEY", "BG_MODAL_SECRET"))
    if not url or not key or not secret:
        raise SystemExit("verify.py needs BG_JUDGE_URL, BG_MODAL_KEY and BG_MODAL_SECRET (the Proxy Token)")
    return Client(url, {"Modal-Key": key, "Modal-Secret": secret})


def main(argv: list[str]) -> int:
    root = validate.ROOT
    ap = argparse.ArgumentParser(description="Grade problems on the live GPU entry.")
    ap.add_argument("--solutions", type=Path, help="this course's directory in course-solutions")
    ap.add_argument("problems", nargs="*", type=Path)
    args = ap.parse_args(argv)
    course = validate.read_yaml(root / "course.yml")[0]["slug"]
    sols = args.solutions or root.parent / "course-solutions" / course
    if not sols.is_dir():
        raise SystemExit(f"no solutions at {sols}; clone bootgpu/course-solutions there or pass --solutions")
    runtime = tomllib.loads((root / "bootgpu.toml").read_text())["runtime"]
    dirs = args.problems or sorted(p for p in (root / "problems").iterdir() if p.is_dir())
    client = client_from_env()
    failed = [d.name for d in dirs if not verify(d, sols / "problems" / d.resolve().name, runtime, client)]
    if failed:
        print(f"verify failed: {' '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
