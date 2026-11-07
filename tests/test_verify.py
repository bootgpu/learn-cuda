import json
import re
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import validate
import verify


def body(rr_status=None, failed=None, status="finished", kind=None):
    tests = [{"name": "example_1", "successful": True}]
    if failed:
        tests.append({"name": failed[0], "successful": False, "kind": failed[1], "message": "Mismatch\nmore"})
    rr = None if rr_status is None else {"status": rr_status, "testResults": tests}
    return {"status": status, "failureKind": kind, "runResult": rr}


class FakeClient:
    """Turns away an unknown runtime (400), else answers by the last word of the submitted code."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []
        self.versions = {"abc1234"}

    def submit(self, payload):
        self.calls.append(dict(payload))
        if payload["runtime"] != "cuda@1":
            return 400, {"error": "unknown_runtime"}
        answer = self.answers[payload["code"].split()[-1]]
        return answer.pop(0) if isinstance(answer, list) else answer


WRONG_ANSWER = (200, body("TESTS_FAILED", ("example_1", "wrong_answer")))
ANSWERS = {
    "ok": (200, body("PASSED")),
    "{}": WRONG_ANSWER,
}


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(verify, "RETRY_SEC", 0)


def test_all_as_declared(bank, sol, capsys):
    client = FakeClient(dict(ANSWERS))
    assert verify.verify(bank, sol, "cuda@1", client)
    assert len(client.calls) == 3 * verify.RUNS
    assert {(c["runtime"], c["language"], c["timeoutSec"]) for c in client.calls} == {("cuda@1", "cuda", 30)}
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", c["requestId"]) for c in client.calls)
    assert len({c["requestId"] for c in client.calls}) == len(client.calls)
    out = capsys.readouterr().out
    assert "map: cuda@1 graded by sandbox deploy abc1234" in out
    assert "map: 9/9 runs as declared" in out


def test_mismatch_fails(bank, sol, capsys):
    assert not verify.verify(bank, sol, "cuda@1", FakeClient(dict(ANSWERS, ok=WRONG_ANSWER)))
    out = capsys.readouterr().out
    assert "FAIL  map/solution run 1" in out and "Mismatch" in out and "map: 3/9" in out


def test_platform_errors_are_retried(bank, sol):
    flaky = [(503, {"error": "busy"}), (200, body(status="system_error", kind="platform")), ANSWERS["ok"]]
    client = FakeClient(dict(ANSWERS, ok=flaky + [ANSWERS["ok"]] * 5))
    assert verify.verify(bank, sol, "cuda@1", client)
    assert len(client.calls) == 3 * verify.RUNS + 2
    assert len({c["requestId"] for c in client.calls}) == len(client.calls)


def test_bad_requests_are_not_retried(bank, sol, capsys):
    client = FakeClient(dict(ANSWERS, ok=[(400, {"error": "bad requestId"})] + [ANSWERS["ok"]] * 5))
    assert not verify.verify(bank, sol, "cuda@1", client)
    assert len(client.calls) == 3 * verify.RUNS
    assert "http 400: bad requestId" in capsys.readouterr().out


def test_course_timeout_is_required(bank, sol, capsys):
    (bank.parent.parent / "bootgpu.toml").write_text('runtime = "cuda@1"\ntimeout_sec = 41\n')
    client = FakeClient(ANSWERS)
    assert not verify.verify(bank, sol, "cuda@1", client)
    assert "timeout_sec must be an integer from 1 to 40" in capsys.readouterr().err
    assert client.calls == []


def test_unknown_runtime_is_not_graded(bank, sol, capsys):
    client = FakeClient(ANSWERS)
    assert not verify.verify(bank, sol, "cuda@9", client)
    assert len(client.calls) == 3 * verify.RUNS
    out = capsys.readouterr().out
    assert "('rejected', 'unknown_runtime', None, None, None)" in out
    assert "map: 0/9 runs as declared" in out


def test_invalid_solutions_are_not_graded(bank, sol, capsys):
    (sol / "variants/cuda/wrong_b.cu").write_text("")
    (sol / "solution/cuda/solution.cu").unlink()
    client = FakeClient(ANSWERS)
    assert not verify.verify(bank, sol, "cuda@1", client)
    err = capsys.readouterr().err
    assert "variants/cuda/wrong_b.cu: unexpected file" in err and "solution/cuda/solution.cu: missing" in err
    assert client.calls == []


def test_variants_are_optional(bank, sol, capsys):
    shutil.rmtree(sol / "variants")
    assert verify.verify(bank, sol, "cuda@1", FakeClient(dict(ANSWERS)))
    assert "map: 6/6 runs as declared" in capsys.readouterr().out


def test_main_reads_solutions_from_course_solutions(bank, sol, monkeypatch):
    monkeypatch.setattr(validate, "ROOT", bank.parent.parent)
    with pytest.raises(SystemExit, match="no solutions at .*course-solutions/learn-demo"):
        verify.main([])
    monkeypatch.setattr(verify, "client_from_env", lambda: FakeClient(dict(ANSWERS)))
    assert verify.main(["--solutions", str(sol.parent.parent)]) == 0


def test_matches():
    assert verify.matches(None, ("finished", None, "TESTS_FAILED", "test_1", "wrong_answer"))
    assert not verify.matches(None, verify.PASSED)
    assert verify.matches(verify.PASSED, verify.PASSED)


class Entry(BaseHTTPRequestHandler):
    """The GPU entry by script: each request takes the next (status, payload, delay) off the server's list.
    A 200's payload is its events (a number in between: a pause in seconds), any other's the body text."""

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.server.requests.append((dict(self.headers), req))
        status, payload, delay = self.server.script.pop(0)
        time.sleep(delay)
        try:
            self.send_response(status)
            self.end_headers()
            if status != 200:
                self.wfile.write(payload.encode())
                return
            for e in payload:
                if isinstance(e, (int, float)):
                    time.sleep(e)
                    continue
                self.wfile.write(json.dumps(e).encode() + b"\n")
                self.wfile.flush()
        except BrokenPipeError:  # the client gave up first
            pass

    def log_message(self, *args):
        pass


@pytest.fixture
def entry():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Entry)
    server.script, server.requests = [], []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    auth = {"Modal-Key": "k", "Modal-Secret": "s"}
    yield server, verify.Client(f"http://127.0.0.1:{server.server_port}", auth)
    server.shutdown()
    server.server_close()


STARTED = {"type": "started", "version": "v1"}
RESULT = {"type": "result", "response": body("PASSED")}
SUB = {"runtime": "cuda@1", "language": "cuda", "timeoutSec": 30, "code": "ok", "requestId": "r1"}
LOST = (500, "modal-http: internal error: status InternalFailure: Server has lost track of input", 0)


def test_client_reads_the_stream_to_its_result(entry):
    server, client = entry
    server.script = [(200, [STARTED, {"type": "heartbeat"}, RESULT], 0)]
    assert client.submit(SUB) == (200, RESULT["response"])
    headers, req = server.requests[0]
    assert (headers["Modal-Key"], headers["Modal-Secret"]) == ("k", "s")
    assert (req["mode"], req["requestId"], req["code"]) == ("submit", "r1", "ok")
    assert 29 < req["deadline"] - time.time() <= 30
    assert client.versions == {"v1"}


def test_client_sends_a_lost_call_again_once_with_the_same_deadline(entry):
    server, client = entry
    server.script = [LOST, (200, [STARTED, RESULT], 0)]
    assert client.submit(SUB)[0] == 200
    assert server.requests[0][1]["deadline"] == server.requests[1][1]["deadline"]
    server.script = [LOST, LOST]
    status, resp = client.submit(SUB)
    assert status == 500 and "lost track of input" in resp["error"]
    assert len(server.requests) == 4


@pytest.mark.parametrize(
    ("script", "want"),
    [
        ([(200, [{"type": "expired"}], 0)], (503, {"error": "busy"})),
        ([(200, [STARTED], 0)], (0, {"error": "the stream ended without a result"})),
        ([(400, '{"error": "unknown_runtime"}', 0)], (400, {"error": "unknown_runtime"})),
        ([(401, "modal-http: missing credentials", 0)], (401, {"error": "modal-http: missing credentials"})),
    ],
)
def test_client_outcomes(entry, script, want):
    server, client = entry
    server.script = script
    assert client.submit(SUB) == want


def test_client_is_busy_without_response_headers_in_time(entry, monkeypatch):
    server, client = entry
    monkeypatch.setattr(verify, "HEADER_SEC", 0.3)
    server.script = [(200, [STARTED, RESULT], 1)]
    assert client.submit(SUB) == (503, {"error": "busy"})


def test_client_gives_up_t_plus_slack_after_started(entry, monkeypatch):
    server, client = entry
    monkeypatch.setattr(verify, "SLACK_SEC", 0.3)
    server.script = [(200, [STARTED, 1, RESULT], 0)]
    assert client.submit(SUB | {"timeoutSec": 0}) == (0, {"error": "no result within T + 0.3 s of started"})
