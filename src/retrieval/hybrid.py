from collections import defaultdict

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bam_circulars"

print("Initializing Hybrid Retriever (Dense + Sparse + RRF)...")

client = QdrantClient(url=QDRANT_URL)

dense_model = TextEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def hybrid_retrieve(
    question: str,
    limit: int = 5,
    candidate_limit: int = 20,
) -> list[dict]:    # Generate query embeddings
    dense_vector = list(dense_model.query_embed([question]))[0].tolist()

    sparse_embedding = list(sparse_model.query_embed([question]))[0]
    sparse_indices = sparse_embedding.indices.tolist()
    sparse_values = sparse_embedding.values.tolist()

    # Search each named vector independently
    dense_hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_vector,
        using="dense",
        limit=candidate_limit,
        with_payload=True,
    ).points

    sparse_hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.SparseVector(
            indices=sparse_indices,
            values=sparse_values,
        ),
        using="sparse",
        limit=candidate_limit,
        with_payload=True,
    ).points

    # Reciprocal Rank Fusion (RRF)
    scores = defaultdict(float)
    hit_by_id = {}

    for hits in (dense_hits, sparse_hits):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.id] += 1.0 / (60 + rank)
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


if __name__ == "__main__":
    question = "Quelles sont les obligations de la banque lors de la clôture d'un compte ?"

    print(f"\nQuestion : {question}")
    results = hybrid_retrieve(question, limit=5)

    for i, result in enumerate(results, start=1):
        print(f"\n--- Résultat {i} | RRF: {result['score']:.4f} ---")
        print(f"Source : {result['source']} | Page : {result['page']}")
        print(result["text"][:500])
