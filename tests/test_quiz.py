import pytest
import validate


def edit(path, old, new):
    text = path.read_text()
    assert old in text, old
    path.write_text(text.replace(old, new, 1))


def check(quiz):
    warnings = []
    return validate.validate(quiz, warnings), warnings


def test_valid(quiz):
    assert check(quiz) == ([], [])
    assert validate.validate_course(quiz.parent.parent) == []


def test_files(quiz):
    (quiz / "quiz.zh.yml").unlink()
    (quiz / "problem.py").write_text("")
    (quiz / "starter").mkdir()
    assert validate.validate(quiz) == [
        "quiz.zh.yml: missing",
        "problem.py: a quiz stage (it has quiz.yml) has no problem.py",
        "starter: a quiz stage (it has quiz.yml) has no starter",
    ]


def test_no_difficulty(quiz):
    chapter = quiz.parent.parent / "chapters/basics/chapter.yml"
    chapter.write_text(chapter.read_text() + "    difficulty: easy\n")
    assert validate.validate(quiz) == ["a quiz stage has no difficulty in chapters/basics/chapter.yml"]


@pytest.mark.parametrize(
    "old, new, error",
    [
        (
            "questions:\n",
            "title: Quiz\nquestions:\n",
            "quiz.yml: must be a mapping with only questions, a list",
        ),
        ("id: threads", "id: Threads", "questions[0]: id must be lowercase letters"),
        ("type: single", "type: choice", "questions[0]: type must be one of single, multiple, blank"),
        ("answer: 2", "answer: 4", "questions[0]: answer must be an option index (0-3)"),
        ("answer: 2", "answers: [2]", "questions[0]: a single question needs answer"),
        ("{ 0: That", "{ 2: That", "questions[0]: feedback.2 is the correct option"),
        ("options: [2, 4, 8, 6]", "options: [2, 4, 8, 4]", "questions[0]: option '4' appears twice"),
        ("options: [2, 4, 8, 6]", "options: [8]", "questions[0]: options must be a list of 2-6 options"),
        ("answers: [2, 0]", "answers: [2, 0, 1]", "questions[1]: answers cannot be every option"),
        ("answers: [2, 0]", "answers: [2, 2]", "questions[1]: answers lists an option twice"),
        (
            "    explanation: Both",
            "    feedback: { 1: No. }\n    explanation: Both",
            "feedback is only for single",
        ),
        ("{ numeric: 4 }", "{ numeric: 4, tolerance: -1 }", "questions[2]: accept must be"),
        ("{ exact: [__global__] }", "{ exact: [] }", "questions[3]: accept must be"),
        (
            "    accept: { numeric",
            "    options: [a, b]\n    accept: { numeric",
            "a blank question cannot have options",
        ),
        ("the number of blocks.", "the <b>number</b> of blocks.", "feedback.0: raw HTML is not allowed"),
        (
            "explanation: Two blocks of four threads.",
            "explanation: ''",
            "questions[0]: explanation must be non-empty",
        ),
    ],
)
def test_structure(quiz, old, new, error):
    edit(quiz / "quiz.yml", old, new)
    errors, _ = check(quiz)
    assert any(error in e for e in errors), errors


def test_question_limits(quiz):
    edit(quiz / "quiz.yml", "id: global-index", "id: threads")
    assert check(quiz)[0] == ["quiz.yml: question id threads appears twice"]
    (quiz / "quiz.yml").write_text("questions: []\n")
    assert check(quiz)[0] == ["quiz.yml: questions must list 1-10 questions"]


@pytest.mark.parametrize(
    "old, new, error",
    [
        (
            "  - id: qualifier\n",
            "  - id: other\n",
            "quiz.zh.yml: questions must have the same ids in the same order",
        ),
        ("options: [2, 4, 8, 6]", "options: [2, 4, 8]", "quiz.zh.yml: threads: 3 options, quiz.yml has 4"),
        (
            "    feedback: { 0: 这是块数。 }\n",
            "",
            "quiz.zh.yml: threads: feedback must cover the same options",
        ),
        (
            "    explanation: 第二个",
            "    options: [a, b]\n    explanation: 第二个",
            "block-size: options are not expected",
        ),
    ],
)
def test_translation(quiz, old, new, error):
    edit(quiz / "quiz.zh.yml", old, new)
    errors, _ = check(quiz)
    assert any(error in e for e in errors), errors


def test_lint_errors(quiz):
    edit(quiz / "quiz.zh.yml", "options: [2, 4, 8, 6]", "options: [2, 4, 8, 以上都不对]")
    assert check(quiz) == (
        ["quiz.zh.yml: threads: option 3 '以上都不对' is a catch-all option; write a real answer"],
        [],
    )
    edit(quiz / "quiz.zh.yml", "以上都不对", "6")
    edit(quiz / "quiz.yml", "options: [2, 4, 8, 6]", "options: [2, 4, 8 threads, 6]")
    errors, _ = check(quiz)
    assert errors == [
        "quiz.yml: threads: the correct option (9 characters) is more than 1.5x every other option"
        " (longest 1); move the reasoning into the explanation"
    ]


def test_lint_warnings(quiz, capsys):
    edit(quiz / "quiz.yml", "options: [2, 4, 8, 6]", "options: [40, 64, 128, 32]")
    edit(quiz / "quiz.zh.yml", "options: [2, 4, 8, 6]", "options: [40, 64, 128, 32]")
    single = "  - id: {0}\n    type: single\n    prompt: P\n    options: [a, b, c]\n    answer: 2\n"
    single_zh = "  - id: {0}\n    prompt: 题\n    options: [a, b, c]\n    explanation: 解\n"
    for qid in ("q2", "q3"):
        (quiz / "quiz.yml").write_text(
            (quiz / "quiz.yml").read_text() + single.format(qid) + "    explanation: E\n"
        )
        (quiz / "quiz.zh.yml").write_text((quiz / "quiz.zh.yml").read_text() + single_zh.format(qid))
    assert check(quiz) == (
        [],
        [
            "quiz.yml: threads: the correct option is the longest one (3 vs 2 characters)",
            "quiz.zh.yml: threads: the correct option is the longest one (3 vs 2 characters)",
            "quiz.yml: 3 single questions have the answer at option 2 (threads, q2, q3); vary the position",
        ],
    )
    assert validate.main([str(quiz)]) == 0
    assert "warning: check/quiz.yml: 3 single questions" in capsys.readouterr().err
