from collections import defaultdict


def reciprocal_rank_fusion(
    dense_hits,
    sparse_hits,
    limit: int = 5,
    k: int = 60,
) -> list[dict]:
    """Combine dense and sparse search results using Reciprocal Rank Fusion."""

    scores = defaultdict(float)
    hit_by_id = {}

    for hits in (dense_hits, sparse_hits):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.id] += 1.0 / (k + rank)
            hit_by_id[hit.id] = hit

    ranked_ids = sorted(scores, key=scores.get, reverse=True)

    results = []

    for point_id in ranked_ids[:limit]:
        hit = hit_by_id[point_id]
        payload = hit.payload or {}

        results.append({
            "id": hit.id,
            "score": scores[point_id],
            "text": payload.get("text", ""),
            "source": payload.get("source", "Inconnu"),
            "page": payload.get("page", "?"),
        })

    return results
