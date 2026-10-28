
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from retrieval.hybrid import hybrid_retrieve

load_dotenv()

llm = ChatGroq(
    temperature=0,
    model_name="openai/gpt-oss-20b",
)


def hybrid_rag(question: str, limit: int = 10) -> dict:
    results = hybrid_retrieve(question, limit=limit)

    contexts = [result["text"] for result in results]
    sources = [
        {
            "source": result["source"],
            "page": result["page"],
        }
        for result in results
    ]

    context_str = "\n\n".join(
        f"--- Extrait de {result['source']} "
        f"(Page {result['page']}) ---\n{result['text']}"
        for result in results
    )

    prompt = PromptTemplate.from_template(
        "Tu es un expert strict en réglementation de Bank Al-Maghrib. "
        "Réponds à la question en utilisant UNIQUEMENT le contexte fourni. "
        "Si l'information n'y figure pas, dis que tu ne sais pas.\n\n"
        "Contexte:\n{context}\n\n"
        "Question: {question}\n"
        "Réponse (cite les documents et pages utilisés):"
    )

    answer = (prompt | llm).invoke(
        {"context": context_str, "question": question}
    ).content

    return {
        "answer": answer,
        "contexts": contexts,
        "sources": sources,
    }


if __name__ == "__main__":
    question = (
        "Quelles sont les conditions qui permettent de combiner "
        "le comité d'audit et le comité des risques en un seul comité ?"
    )

    result = hybrid_rag(question, limit=10)

    print("\nRéponse :")
    print(result["answer"])

    print("\nSources :")
    for source in result["sources"]:
        print(f"- {source['source']} | Page {source['page']}")
