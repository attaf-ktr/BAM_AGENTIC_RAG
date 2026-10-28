
import json
import math
import sys
import time
from pathlib import Path

from groq import RateLimitError
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from datasets import Dataset

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# ============================================================
# PROJECT PATHS AND IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

# Allows imports from src/retrieval when running:
# python src/eval/run_eval.py
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval.naive import naive_rag

GOLDEN_FILE = PROJECT_ROOT / "data" / "eval" / "golden.jsonl"
RESULTS_FILE = PROJECT_ROOT / "results" / "baseline.json"
CACHE_FILE = PROJECT_ROOT / "results" / "baseline_cache.json"

RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

print("Initialisation de l'evaluation RAG...")


# ============================================================
# JSON SAVE
# ============================================================

def save_json(path, data):
    """Writes JSON safely through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=str)

    temp_path.replace(path)


# ============================================================
# LOAD GOLDEN DATASET
# ============================================================

def load_golden_dataset():
    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {GOLDEN_FILE}"
        )

    questions = []

    with GOLDEN_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON invalide ligne {line_number} de {GOLDEN_FILE}"
                ) from exc

            question = (
                item.get("question")
                or item.get("user_input")
                or item.get("query")
            )

            reference = (
                item.get("reference")
                or item.get("reference_answer")
                or item.get("ground_truth")
                or item.get("answer")
)

            if not question:
                raise ValueError(
                    f"Question manquante ligne {line_number}."
                )

            if not reference:
                raise ValueError(
                    f"Réponse de référence manquante ligne {line_number}."
                )

            questions.append({
                "question": question,
                "reference": reference,
            })

    if not questions:
        raise ValueError("Le dataset golden est vide.")

    return questions


# ============================================================
# CALL NAIVE RAG WITH RATE-LIMIT RETRIES
# ============================================================

def call_naive_rag_with_retry(question, max_attempts=8):
    for attempt in range(1, max_attempts + 1):
        try:
            return naive_rag(question)

        except RateLimitError:
            if attempt == max_attempts:
                print(
                    f"Groq rate limit persistant après "
                    f"{max_attempts} tentatives."
                )
                raise

            wait_seconds = min(15 * attempt, 90)

            print(
                f"Groq rate limit (tentative {attempt}/{max_attempts}). "
                f"Attente de {wait_seconds} secondes."
            )
            time.sleep(wait_seconds)


# ============================================================
# NORMALIZE RAG RESULT
# ============================================================

def normalize_rag_result(result):
    """
    naive_rag returns:
    {
        "answer": str,
        "contexts": list[str],
        "sources": list[dict]
    }
    """
    if not isinstance(result, dict):
        raise TypeError(
            f"Format de retour inattendu : {type(result).__name__}"
        )

    answer = result.get("answer")
    contexts = result.get("contexts") or []

    if not answer:
        raise ValueError("naive_rag n'a pas renvoyé de réponse.")

    if isinstance(contexts, str):
        contexts = [contexts]

    normalized_contexts = []

    for context in contexts:
        if isinstance(context, str):
            normalized_contexts.append(context)
        elif hasattr(context, "page_content"):
            normalized_contexts.append(context.page_content)
        elif isinstance(context, dict):
            text = (
                context.get("page_content")
                or context.get("text")
                or context.get("content")
            )
            if text:
                normalized_contexts.append(str(text))

    return {
        "answer": str(answer),
        "contexts": [
            context for context in normalized_contexts if context
        ],
    }


# ============================================================
# METRIC HELPERS
# ============================================================

def finite_number(value):
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


# ============================================================
# EVALUATE ONE QUESTION WITH RAGAS
# ============================================================

def evaluate_one_question(row, judge_llm, embeddings):
    dataset = Dataset.from_dict({
        "question": [row["question"]],
        "answer": [row["answer"]],
        "contexts": [row["contexts"]],
        "reference": [row["reference"]],
    })

    run_config = RunConfig(
        max_workers=1,
        max_retries=3,
        max_wait=90,
        timeout=180,
        log_tenacity=True,
    )

    for attempt in range(1, 4):
        try:
            result = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
                llm=judge_llm,
                embeddings=embeddings,
                run_config=run_config,
                raise_exceptions=False,
                show_progress=False,
            )

            result_dict = result.to_pandas().iloc[0].to_dict()

            scores = {}
            missing = []

            for metric in METRIC_NAMES:
                score = finite_number(result_dict.get(metric))
                scores[metric] = score

                if score is None:
                    missing.append(metric)

            return scores, missing

        except RateLimitError:
            if attempt == 3:
                break

            wait_seconds = min(30 * attempt, 120)
            print(
                f"Rate limit pendant RAGAS "
                f"(tentative {attempt}/3). "
                f"Pause de {wait_seconds} secondes."
            )
            time.sleep(wait_seconds)

        except Exception as exc:
            print(f"Erreur RAGAS : {type(exc).__name__}: {exc}")
            break

    return (
        {metric: None for metric in METRIC_NAMES},
        METRIC_NAMES.copy(),
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation():
    questions = load_golden_dataset()
    print(f"{len(questions)} questions chargées.")

    # Load cache to avoid regenerating existing answers.
    cache = {}

    if CACHE_FILE.exists():
        try:
            with CACHE_FILE.open("r", encoding="utf-8") as file:
                cache = json.load(file)

            print(f"Cache chargé : {len(cache)} entrées.")

        except (json.JSONDecodeError, OSError):
            print("Cache illisible : création d'un nouveau cache.")
            cache = {}

    # Generate answers using the baseline RAG.
    print("\nGénération des réponses RAG...")
    rows = []

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        reference = item["reference"]
        cache_key = str(index)

        cached = cache.get(cache_key, {})

        if (
            cached.get("question") == question
            and cached.get("reference") == reference
            and cached.get("answer")
        ):
            print(f"[{index}/{len(questions)}] Réponse récupérée du cache")
            answer = cached["answer"]
            contexts = cached.get("contexts", [])

        else:
            print(f"[{index}/{len(questions)}] {question[:90]}")

            raw_result = call_naive_rag_with_retry(question)
            normalized = normalize_rag_result(raw_result)

            answer = normalized["answer"]
            contexts = normalized["contexts"]

            cache[cache_key] = {
                "question": question,
                "reference": reference,
                "answer": answer,
                "contexts": contexts,
            }
            save_json(CACHE_FILE, cache)

        rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "reference": reference,
        })

    # This measures context coverage, not retrieval relevance.
    context_coverage = (
        sum(bool(row["contexts"]) for row in rows) / len(rows)
        if rows else 0.0
    )

    # Configure the RAGAS judge.
    print("\nInitialisation du juge RAGAS...")

    judge_llm = ChatGroq(
        temperature=0,
        model="openai/gpt-oss-120b",
        max_retries=3,
        timeout=180,
    )

    embeddings = FastEmbedEmbeddings()
    question_results = []

    # Evaluate questions individually to reduce request load.
    for index, row in enumerate(rows, start=1):
        print(f"\n[{index}/{len(rows)}] Évaluation en cours...")

        scores, missing = evaluate_one_question(
            row=row,
            judge_llm=judge_llm,
            embeddings=embeddings,
        )

        question_results.append({
            "question": row["question"],
            "answer": row["answer"],
            "contexts": row["contexts"],
            "reference": row["reference"],
            "scores": scores,
            "missing_metrics": missing,
        })

        print(
            "Scores : "
            + " | ".join(
                f"{metric}={scores[metric]}"
                for metric in METRIC_NAMES
            )
        )

        if missing:
            print(f"Métriques manquantes : {', '.join(missing)}")

        # Save partial results after every question.
        summary = {}

        for metric in METRIC_NAMES:
            valid_scores = [
                result["scores"][metric]
                for result in question_results
                if result["scores"][metric] is not None
            ]

            summary[f"ragas_{metric}"] = (
                sum(valid_scores) / len(valid_scores)
                if valid_scores else None
            )
            summary[f"ragas_{metric}_valid_count"] = len(valid_scores)

        partial_result = {
            "num_questions": len(rows),
            "evaluated_questions": len(question_results),
            "context_coverage": context_coverage,
            "summary": summary,
            "question_results": question_results,
        }

        save_json(RESULTS_FILE, partial_result)

        if index < len(rows):
            time.sleep(15)

    # Final summary.
    summary = {}

    for metric in METRIC_NAMES:
        valid_scores = [
            result["scores"][metric]
            for result in question_results
            if result["scores"][metric] is not None
        ]

        summary[f"ragas_{metric}"] = (
            sum(valid_scores) / len(valid_scores)
            if valid_scores else None
        )
        summary[f"ragas_{metric}_valid_count"] = len(valid_scores)

    final_result = {
        "num_questions": len(rows),
        "context_coverage": context_coverage,
        "summary": summary,
        "question_results": question_results,
    }

    save_json(RESULTS_FILE, final_result)

    print("\n" + "=" * 55)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("=" * 55)
    print(f"Questions : {len(rows)}")
    print(f"Questions avec contexte : {context_coverage:.2%}")

    for metric in METRIC_NAMES:
        score = summary[f"ragas_{metric}"]
        valid = summary[f"ragas_{metric}_valid_count"]

        if score is None:
            print(f"{metric} : aucun score valide ({valid}/{len(rows)})")
        else:
            print(
                f"{metric} : {score:.4f} "
                f"({valid}/{len(rows)} scores valides)"
            )

    print("=" * 55)
    print(f"Résultats : {RESULTS_FILE}")
    print(f"Cache : {CACHE_FILE}")


if __name__ == "__main__":
    run_evaluation()
