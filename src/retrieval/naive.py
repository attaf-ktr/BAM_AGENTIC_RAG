import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# 1. Chargement des variables d'environnement (.env)
load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bam_circulars"

print("⚙️ Initialisation du RAG Naïf (Baseline)...")
client = QdrantClient(url=QDRANT_URL)
dense_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# Utilisation du dernier modèle Groq recommandé
llm = ChatGroq(temperature=0, model_name="openai/gpt-oss-20b")

def naive_rag(question: str) -> dict:
    query_vector = list(dense_model.embed([question]))[0].tolist()
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="dense",
        limit=5,
    ).points

    contexts, sources = [], []
    for hit in hits:
        text = hit.payload.get("text", "")
        source = hit.payload.get("source", "Inconnu")
        page = hit.payload.get("page", "?")
        contexts.append(text)
        sources.append({"source": source, "page": page})

    context_str = "\n\n".join(
        f"--- Extrait de {s['source']} (Page {s['page']}) ---\n{c}"
        for c, s in zip(contexts, sources)
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

    return {"answer": answer, "contexts": contexts, "sources": sources}

if __name__ == "__main__":
    # Test interactif rapide
    test_question = "Quelles sont les obligations de la banque lors de la clôture d'un compte ?"
    print(f"\n❓ Question : {test_question}")
    
    result = naive_rag(test_question)
    print(result["answer"])
    print(result["sources"])