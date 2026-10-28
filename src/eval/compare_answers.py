import argparse
import json
import re
import unicodedata
from pathlib import Path

from retrieval.naive import naive_rag
from retrieval.hybrid_rag import hybrid_rag

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_FILE = ROOT / "data" / "eval" / "golden.jsonl"
OUTPUT_FILE = ROOT / "results" / "answer_comparison.jsonl"
REPORT_FILE = ROOT / "results" / "answer_comparison_report.json"


def normalize(value):
    value = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", value).strip().casefold()


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def expected_source_found(expected, sources):
    if normalize(expected) in ("", "aucun", "none"):
        return None

    expected_name = normalize(Path(expected).name)
    return any(
        normalize(Path(s.get("source", "")).name) == expected_name
        for s in sources
    )


def expected_page_found(expected_source, expected_page, sources):
    if normalize(expected_source) in ("", "aucun", "none"):
        return None

    expected_name = normalize(Path(expected_source).name)
    return any(
        normalize(Path(s.get("source", "")).name) == expected_name
        and str(s.get("page", "")).strip() == str(expected_page).strip()
        for s in sources
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["both", "naive", "hybrid"],
        default="both",
        help="Which pipeline to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N questions; 0 means all",
    )
    args = parser.parse_args()

    golden = load_jsonl(GOLDEN_FILE)
    if args.limit > 0:
        golden = golden[:args.limit]

    methods = (
        ["naive", "hybrid"] if args.method == "both" else [args.method]
    )

    existing = load_jsonl(OUTPUT_FILE) if OUTPUT_FILE.exists() else []
    completed = {
        (row["method"], row["question"])
        for row in existing
    }

    for index, item in enumerate(golden, start=1):
        question = item["question"]

        for method in methods:
            if (method, question) in completed:
                print(f"[{index}/{len(golden)}] {method}: cached, skipping")
                continue

            print(f"[{index}/{len(golden)}] Running {method}: {question}")

            rag = naive_rag if method == "naive" else hybrid_rag
            result = rag(question)

            row = {
                "index": index,
                "method": method,
                "question": question,
                "reference_answer": item.get("reference_answer", ""),
                "expected_source": item.get("source_doc", "Aucun"),
                "expected_page": item.get("source_page", 0),
                "answer": result.get("answer", ""),
                "contexts": result.get("contexts", []),
                "sources": result.get("sources", []),
            }

            row["expected_document_retrieved"] = expected_source_found(
                row["expected_source"], row["sources"]
            )
            row["expected_page_retrieved"] = expected_page_found(
                row["expected_source"], row["expected_page"], row["sources"]
            )

            existing.append(row)
            completed.add((method, question))

            # Save after every successful answer to support resuming.
            save_jsonl(OUTPUT_FILE, existing)
            print(f"Saved {method} answer.")

    selected_questions = {item["question"] for item in golden}
    selected_rows = [
        row for row in existing
        if row["question"] in selected_questions
    ]

    report = {
        "questions_evaluated": len(golden),
        "methods": {},
        "note": (
            "Document/page retrieval matches are citation proxies, "
            "not a full assessment of answer faithfulness or correctness."
        ),
    }

    for method in methods:
        rows = [r for r in selected_rows if r["method"] == method]
        answerable = [
            r for r in rows
            if normalize(r["expected_source"]) not in ("", "aucun", "none")
        ]

        def rate(field):
            values = [r[field] for r in answerable if r[field] is not None]
            if not values:
                return None
            return round(sum(bool(v) for v in values) / len(values), 4)

        report["methods"][method] = {
            "answers_saved": len(rows),
            "answerable_questions": len(answerable),
            "expected_document_retrieved_rate": rate(
                "expected_document_retrieved"
            ),
            "expected_page_retrieved_rate": rate(
                "expected_page_retrieved"
            ),
            "average_answer_characters": (
                round(sum(len(r["answer"]) for r in rows) / len(rows))
                if rows else 0
            ),
        }

    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nEvaluation summary")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nDetailed answers: {OUTPUT_FILE}")
    print(f"Summary report:   {REPORT_FILE}")


if __name__ == "__main__":
    main()
