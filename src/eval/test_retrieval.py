
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from retrieval.hybrid import hybrid_retrieve

client = QdrantClient(url="http://localhost:6333")
dense_model = TextEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

GOLDEN = ROOT / "data/eval/golden.jsonl"
OUTPUT = ROOT / "results/retrieval_comparison.json"

NAIVE_K = 5
HYBRID_LIMITS = [5, 10]


def normalize(value):
    value = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", value).strip().casefold()


def retrieve_naive(question):
    vector = list(dense_model.embed([question]))[0].tolist()

    hits = client.query_points(
        collection_name="bam_circulars",
        query=vector,
        using="dense",
        limit=NAIVE_K,
        with_payload=True,
    ).points

    return [
        {
            "source": (h.payload or {}).get("source", "Inconnu"),
            "page": (h.payload or {}).get("page", "?"),
        }
        for h in hits
    ]


def score(results, expected_doc, expected_page):
    doc_ranks, exact_ranks = [], []

    for rank, result in enumerate(results, start=1):
        doc_match = (
            normalize(Path(str(result["source"])).name)
            == normalize(Path(str(expected_doc)).name)
        )

        if doc_match:
            doc_ranks.append(rank)

            try:
                page_match = int(result["page"]) == int(expected_page)
            except (ValueError, TypeError):
                page_match = (
                    normalize(result["page"]) == normalize(expected_page)
                )

            if page_match:
                exact_ranks.append(rank)

    return {
        "doc_hit": bool(doc_ranks),
        "page_hit": bool(exact_ranks),
        "doc_rr": 1 / min(doc_ranks) if doc_ranks else 0,
        "page_rr": 1 / min(exact_ranks) if exact_ranks else 0,
        "doc_rank": min(doc_ranks) if doc_ranks else None,
        "page_rank": min(exact_ranks) if exact_ranks else None,
    }


def summarize(scores, total_questions):
    count = len(scores)

    if count == 0:
        return {
            "questions": 0,
            "document_hit_rate": 0,
            "exact_page_hit_rate": 0,
            "document_mrr": 0,
            "exact_page_mrr": 0,
        }

    return {
        "questions": count,
        "total_golden_questions": total_questions,
        "document_hit_rate": sum(x["doc_hit"] for x in scores) / count,
        "exact_page_hit_rate": sum(x["page_hit"] for x in scores) / count,
        "document_mrr": sum(x["doc_rr"] for x in scores) / count,
        "exact_page_mrr": sum(x["page_rr"] for x in scores) / count,
    }


def print_summary(name, summary):
    print(f"\n{name}")
    print(f"  Answerable questions: {summary['questions']}")

    for metric in (
        "document_hit_rate",
        "exact_page_hit_rate",
        "document_mrr",
        "exact_page_mrr",
    ):
        value = summary[metric]
        print(f"  {metric}: {value:.3f} ({value * 100:.1f}%)")


def main():
    with GOLDEN.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    if not questions:
        raise ValueError(f"No questions found in {GOLDEN}")

    answerable = [
        item for item in questions
        if normalize(item.get("source_doc")) != "aucun"
    ]
    unanswerable_count = len(questions) - len(answerable)

    all_scores = {
        "naive_top5": [],
        **{f"hybrid_top{k}": [] for k in HYBRID_LIMITS},
    }
    details = []

    print(f"Total golden questions: {len(questions)}")
    print(f"Answerable: {len(answerable)}")
    print(f"Unanswerable: {unanswerable_count}")
    print("Evaluating retrieval only; no LLM/Groq calls.")

    for i, item in enumerate(questions, start=1):
        question = item["question"]
        expected_doc = item["source_doc"]
        expected_page = item["source_page"]
        is_answerable = normalize(expected_doc) != "aucun"

        naive_results = retrieve_naive(question)
        naive_metrics = (
            score(naive_results, expected_doc, expected_page)
            if is_answerable else None
        )

        hybrid_data = {}

        for k in HYBRID_LIMITS:
            results = hybrid_retrieve(
                question,
                limit=k,
                candidate_limit=20,
            )

            metrics = (
                score(results, expected_doc, expected_page)
                if is_answerable else None
            )

            hybrid_data[str(k)] = {
                "metrics": metrics,
                "results": results,
            }

        if is_answerable:
            all_scores["naive_top5"].append(naive_metrics)

            for k in HYBRID_LIMITS:
                all_scores[f"hybrid_top{k}"].append(
                    hybrid_data[str(k)]["metrics"]
                )

        details.append({
            "question": question,
            "expected_source": expected_doc,
            "expected_page": expected_page,
            "answerable": is_answerable,
            "naive_top5": {
                "metrics": naive_metrics,
                "results": naive_results,
            },
            "hybrid": hybrid_data,
        })

        if is_answerable:
            print(f"\n[{i}/{len(questions)}] {question}")
            print(f"  Expected: {expected_doc} | page {expected_page}")
            print(f"  Naive top 5 page rank: {naive_metrics['page_rank']}")

            for k in HYBRID_LIMITS:
                metrics = hybrid_data[str(k)]["metrics"]
                print(f"  Hybrid top {k} page rank: {metrics['page_rank']}")
        else:
            print(f"\n[{i}/{len(questions)}] Unanswerable: skipped in hit-rate metrics")

    summaries = {
        name: summarize(values, len(questions))
        for name, values in all_scores.items()
    }

    print("\n========== SUMMARY ==========")
    for name, summary in summaries.items():
        print_summary(name, summary)

    print("\n========== TOP 5 VS TOP 10 ==========")
    for metric in ("document_hit_rate", "exact_page_hit_rate", "document_mrr", "exact_page_mrr"):
        top5 = summaries["hybrid_top5"][metric]
        top10 = summaries["hybrid_top10"][metric]
        delta = top10 - top5
        print(f"{metric}: {top5:.3f} -> {top10:.3f} (change {delta:+.3f})")

    print("\n========== QUESTIONS IMPROVED AT TOP 10 ==========")
    improved = 0

    for item in details:
        if not item["answerable"]:
            continue

        m5 = item["hybrid"]["5"]["metrics"]
        m10 = item["hybrid"]["10"]["metrics"]

        if (
            (not m5["page_hit"] and m10["page_hit"])
            or (not m5["doc_hit"] and m10["doc_hit"])
        ):
            improved += 1
            print(f"- {item['question']}")
            print(f"  Top 5: document={m5['doc_hit']}, page={m5['page_hit']}")
            print(f"  Top 10: document={m10['doc_hit']}, page={m10['page_hit']}")

    if improved == 0:
        print("No questions gained a document or exact-page hit at top 10.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summaries,
                "details": details,
                "settings": {
                    "naive_limit": NAIVE_K,
                    "hybrid_limits": HYBRID_LIMITS,
                    "candidate_limit": 20,
                    "unanswerable_questions_excluded_from_metrics": True,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nDetails saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
