import os
import uuid
import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, SparseVectorParams
from fastembed import TextEmbedding, SparseTextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Configuration
PDF_DIR = "data/raw_pdfs"
COLLECTION_NAME = "bam_circulars"
QDRANT_URL = "http://localhost:6333"

# 2. Initialisation des modèles
print("⚙️ Chargement des modèles (Dense et Sparse)...")
dense_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# 3. Initialisation de Qdrant
client = QdrantClient(url=QDRANT_URL)

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams()}
)
print(f"📁 Collection '{COLLECTION_NAME}' recréée dans Qdrant.")

def extract_text_from_pdfs(directory):
    """Extrait le texte des PDF avec PyMuPDF en gardant le numéro de page."""
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            doc = fitz.open(filepath)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text").strip()
                if text:
                    docs.append({
                        "text": text,
                        "metadata": {"source": filename, "page": page_num + 1}
                    })
    return docs

def process_and_ingest():
    print("📄 Extraction du texte des circulaires BAM...")
    raw_docs = extract_text_from_pdfs(PDF_DIR)
    
    if not raw_docs:
        print("⚠️ Aucun PDF trouvé dans data/raw_pdfs/")
        return

    # Découpage (400 tokens, 15% d'overlap environ)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = []
    
    print("✂️ Découpage en chunks...")
    for doc in raw_docs:
        splits = splitter.split_text(doc["text"])
        for split in splits:
            chunks.append({"text": split, "metadata": doc["metadata"]})
            
    print(f"📊 {len(chunks)} chunks générés. Début de l'encodage hybride...")
    
    # Extraction des textes purs pour FastEmbed
    texts = [chunk["text"] for chunk in chunks]
    
    # Génération des vecteurs
    dense_embeddings = list(dense_model.embed(texts))
    sparse_embeddings = list(sparse_model.embed(texts))
    
    points = []
    for i, chunk in enumerate(chunks):
        point_id = str(uuid.uuid4())
        
        # Format attendu par Qdrant pour les vecteurs épars
        sparse_vector = {
            "indices": sparse_embeddings[i].indices.tolist(),
            "values": sparse_embeddings[i].values.tolist()
        }
        
        points.append(
            PointStruct(
                id=point_id,
                vector={
                    "dense": dense_embeddings[i].tolist(),
                    "sparse": sparse_vector
                },
                payload={"text": chunk["text"], **chunk["metadata"]}
            )
        )
        
    print("💾 Insertion dans Qdrant...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print("✅ Ingestion hybride terminée avec succès !")

if __name__ == "__main__":
    process_and_ingest()