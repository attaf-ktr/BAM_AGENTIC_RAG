import os
import json
import random
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bam_circulars"
OUTPUT_FILE = "data/eval/golden.jsonl"
NUM_QUESTIONS = 15 # Nombre de questions à générer pour commencer

print("⚙️ Initialisation du générateur de Golden Dataset...")
client = QdrantClient(url=QDRANT_URL)

# On garde le même modèle Groq que pour la baseline
llm = ChatGroq(temperature=0.2, model_name="openai/gpt-oss-20b")

def get_random_chunks(limit=20):
    """Récupère un échantillon aléatoire de chunks depuis Qdrant"""
    # scroll() permet de parcourir la base sans vecteur de recherche
    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=100, # On en prend 100 pour mélanger
        with_payload=True,
        with_vectors=False
    )
    return random.sample(records, min(limit, len(records)))

def generate_golden_dataset():
    chunks = get_random_chunks(NUM_QUESTIONS)
    
    prompt = PromptTemplate.from_template(
        "Tu es un expert en réglementation bancaire marocaine (Bank Al-Maghrib). "
        "À partir du texte suivant, génère UNE question claire et précise qu'un professionnel "
        "pourrait poser, et sa réponse de référence basée UNIQUEMENT sur ce texte.\n\n"
        "Texte :\n{context}\n\n"
        "Réponds STRICTEMENT sous forme d'objet JSON valide avec les clés 'question' et 'reference_answer'. "
        "N'ajoute aucun texte avant ou après le JSON."
    )
    
    chain = prompt | llm | JsonOutputParser()
    
    print(f"🤖 Génération de {len(chunks)} paires Q/A en cours...")
    
    # On vide ou crée le fichier
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        pass

    success_count = 0
    for i, chunk in enumerate(chunks):
        text = chunk.payload.get("text", "")
        source = chunk.payload.get("source", "Inconnu")
        page = chunk.payload.get("page", 0)
        
        # Ignorer les chunks trop petits ou vides
        if len(text) < 100:
            continue
            
        try:
            print(f"  [{i+1}/{len(chunks)}] Génération pour {source} (Page {page})...")
            result = chain.invoke({"context": text})
            
            # Assemblage de l'objet final
            golden_entry = {
                "question": result.get("question"),
                "reference_answer": result.get("reference_answer"),
                "source_doc": source,
                "source_page": page
            }
            
            # Sauvegarde immédiate (Append)
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(golden_entry, ensure_ascii=False) + '\n')
            
            success_count += 1
            
        except Exception as e:
            print(f"  ⚠️ Échec de génération pour le chunk {i+1}: {e}")
            continue

    print(f"\n✅ Terminé ! {success_count} questions générées et sauvegardées dans {OUTPUT_FILE}.")

if __name__ == "__main__":
    generate_golden_dataset()