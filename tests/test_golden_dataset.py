import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "eval" / "golden.jsonl"


def load_dataset():
    with GOLDEN.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def test_dataset_is_not_empty():
    assert GOLDEN.is_file()
    assert load_dataset()


def test_questions_have_required_fields():
    required = {
        "question",
        "reference_answer",
        "source_doc",
        "source_page",
    }

    for index, item in enumerate(load_dataset(), start=1):
        missing = required - item.keys()
        assert not missing, f"Line {index}: missing {missing}"

        assert isinstance(item["question"], str)
        assert item["question"].strip()


def test_dataset_contains_answerable_and_unanswerable_questions():
    questions = load_dataset()

    answerable = [
        q for q in questions
        if q["source_doc"].strip().casefold() != "aucun"
    ]
    unanswerable = [
        q for q in questions
        if q["source_doc"].strip().casefold() == "aucun"
    ]

    assert answerable
    assert unanswerable
