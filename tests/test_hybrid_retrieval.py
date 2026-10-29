
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval.hybrid import hybrid_retrieve

GOLDEN = ROOT / "data" / "eval" / "golden.jsonl"
TOP_K = 10
NUM_QUESTIONS = 15


def load_answerable_questions():
    with GOLDEN.open(encoding="utf-8") as file:
        questions = [json.loads(line) for line in file if line.strip()]

    return [
        item for item in questions
        if str(item.get("source_doc", "")).strip().casefold() != "aucun"
    ][:NUM_QUESTIONS]


@pytest.mark.parametrize(
    "item",
    load_answerable_questions(),
    ids=lambda item: item["question"][:60],
)
def test_hybrid_retrieves_expected_document_and_page(item):
    results = hybrid_retrieve(item["question"], limit=TOP_K)

    expected_doc = Path(item["source_doc"]).name.casefold()
    expected_page = str(item["source_page"]).strip()

    assert results, f"No results returned for: {item['question']}"

    matching_document = [
        result for result in results
        if Path(str(result["source"])).name.casefold() == expected_doc
    ]

    assert matching_document, (
        f"Expected document {expected_doc!r} not found in top {TOP_K}. "
        f"Retrieved: {[r['source'] for r in results]}"
    )

    matching_page = [
        result for result in matching_document
        if str(result["page"]).strip() == expected_page
    ]

    assert matching_page, (
        f"Expected page {expected_page!r} from {expected_doc!r} "
        f"not found in top {TOP_K}. "
        f"Retrieved: {[(r['source'], r['page']) for r in results]}"
    )
