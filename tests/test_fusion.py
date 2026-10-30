import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval.fusion import reciprocal_rank_fusion
def make_hit(point_id, source="doc.pdf", page=1):
    return SimpleNamespace(
        id=point_id,
        payload={
            "text": f"Text from {point_id}",
            "source": source,
            "page": page,
        },
    )


def test_fusion_combines_dense_and_sparse_results():
    dense = [make_hit("A"), make_hit("B")]
    sparse = [make_hit("B"), make_hit("C")]

    results = reciprocal_rank_fusion(dense, sparse, limit=3)

    assert len(results) == 3
    assert results[0]["id"] == "B"


def test_fusion_respects_limit():
    dense = [make_hit("A"), make_hit("B"), make_hit("C")]

    results = reciprocal_rank_fusion(dense, [], limit=2)

    assert len(results) == 2


def test_fusion_returns_expected_payload():
    dense = [make_hit("A", source="circular.pdf", page=12)]

    results = reciprocal_rank_fusion(dense, [])

    assert results[0]["source"] == "circular.pdf"
    assert results[0]["page"] == 12
    assert results[0]["text"] == "Text from A"


def test_fusion_handles_empty_results():
    assert reciprocal_rank_fusion([], []) == []
