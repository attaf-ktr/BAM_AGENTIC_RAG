# BAM Agentic RAG

A Retrieval-Augmented Generation (RAG) system for querying Bank Al-Maghrib regulatory circulars. The project explores and evaluates dense retrieval against hybrid dense + sparse retrieval to improve the relevance of regulatory document retrieval and answer generation.

## Overview

The system retrieves relevant passages from regulatory documents and uses a language model to generate answers grounded in the retrieved context.

The project currently includes:

- A **naive RAG baseline** using dense vector retrieval.
- A **hybrid RAG pipeline** combining dense and sparse retrieval.
- **Reciprocal Rank Fusion (RRF)** to combine retrieval rankings.
- A golden evaluation dataset containing questions, reference answers, and expected source documents and pages.
- Evaluation scripts for comparing retrieval results and generated answers.

## Architecture

### 1. Document ingestion

The ingestion pipeline in `src/ingest/parser.py` processes regulatory PDFs and prepares content for storage in Qdrant.

Source PDFs are expected under `data/raw_pdfs/`. These files are excluded from Git.

### 2. Vector database

The project uses [Qdrant](https://qdrant.tech/) as its vector database. The collection used by the retrieval pipelines is `bam_circulars`.

The retrieval code connects to Qdrant at `http://localhost:6333`.

### 3. Retrieval strategies

**Naive RAG — `src/retrieval/naive.py`**

- Uses multilingual MiniLM embeddings.
- Retrieves the top 5 chunks using dense similarity search.
- Passes the retrieved context to the language model.

**Hybrid RAG — `src/retrieval/hybrid.py`**

- Uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` through FastEmbed for dense embeddings.
- Uses `Qdrant/bm25` for sparse retrieval.
- Retrieves candidates from both search methods.
- Combines rankings using Reciprocal Rank Fusion (RRF).
- Returns the highest-ranked chunks with their source documents and page metadata.

The hybrid answer-generation pipeline is implemented in `src/retrieval/hybrid_rag.py`. It retrieves 10 chunks by default and can accept a configurable retrieval limit.

### 4. Answer generation

Both RAG pipelines use Groq through LangChain with the `openai/gpt-oss-20b` model.

The prompt instructs the model to answer using only the supplied context and to acknowledge when the information is not available. Answers are requested with document and page citations.

These instructions help constrain responses but do not guarantee that every generated answer is correct or fully supported.

## Project structure

```text
bam-agentic-rag/
├── .github/
│   └── workflows/
│       └── eval.yml
├── data/
│   └── eval/
│       └── golden.jsonl
├── src/
│   ├── eval/
│   │   ├── compare_answers.py
│   │   ├── compare_hybrid_limits.py
│   │   ├── generate_golden.py
│   │   ├── run_eval.py
│   │   └── test_retrieval.py
│   ├── ingest/
│   │   └── parser.py
│   └── retrieval/
│       ├── hybrid.py
│       ├── hybrid_rag.py
│       └── naive.py
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Requirements

- Python
- Docker and Docker Compose
- A Groq API key for answer generation
- Access to the required embedding models
- A populated Qdrant collection for retrieval

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/attaf-ktr/BAM_AGENTIC_RAG.git
cd BAM_AGENTIC_RAG
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows, activate the environment with:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file based on `.env.example`.

At minimum, configure:

```env
GROQ_API_KEY=your_groq_api_key
```

Other integrations may require additional credentials. Never commit real API keys or secrets.

### 5. Start Qdrant

```bash
docker compose up -d
```

Qdrant's HTTP API should then be accessible at `http://localhost:6333`.

### 6. Ingest the documents

Place the required regulatory PDFs in `data/raw_pdfs/`, then run:

```bash
PYTHONPATH=src python src/ingest/parser.py
```

**Warning:** inspect the ingestion script before rerunning it against an existing database. It may recreate the collection and replace existing indexed data.

## Running the RAG pipelines

Run commands from the project root with the virtual environment activated.

### Naive RAG

```bash
PYTHONPATH=src python src/retrieval/naive.py
```

### Hybrid RAG

```bash
PYTHONPATH=src python src/retrieval/hybrid_rag.py
```

### Test hybrid retrieval

```bash
PYTHONPATH=src python src/retrieval/hybrid.py
```

The retrieval script prints ranked passages together with their RRF scores, source documents, and page metadata.

## Evaluation

The golden dataset is stored in `data/eval/golden.jsonl`. It contains evaluation questions, reference answers, expected source documents, and expected pages.

Available evaluation scripts:

| Script | Purpose |
|---|---|
| `test_retrieval.py` | Compare retrieval results and retrieval metrics |
| `compare_answers.py` | Compare generated answers from naive and hybrid RAG |
| `compare_hybrid_limits.py` | Compare hybrid answer generation with different retrieval limits |
| `run_eval.py` | Run the configured RAGAS evaluation |
| `generate_golden.py` | Generate the golden evaluation dataset |

Run an evaluation script from the repository root, for example:

```bash
PYTHONPATH=src python src/eval/test_retrieval.py
```

Answer-generation and RAGAS evaluations may consume API quota. Check the script's options before running a full evaluation.

### Evaluation metrics

The project examines metrics such as:

- **Document hit rate:** whether the expected source document appears among retrieved results.
- **Exact-page hit rate:** whether the expected document and page are retrieved.
- **Mean Reciprocal Rank (MRR):** how highly the expected result ranks.
- **Answer quality:** assessed separately from retrieval, using answer comparisons and configured evaluation metrics.

Retrieving the expected document or page is useful evidence about retrieval performance, but it does not by itself prove that a generated answer is correct, faithful, or complete.

## Current limitations

- Retrieval depends on the documents indexed in Qdrant.
- Generated answers may still contain unsupported details despite context-grounded prompting.
- Evaluation results depend on the golden dataset and selected metrics.
- The full evaluation pipeline may require external API credentials and additional configuration.

## Roadmap

- Automate lightweight tests with GitHub Actions.
- Improve documentation and reproducibility.
- Continue comparing dense and hybrid retrieval.
- Evaluate the effect of retrieval limits on answer quality.
- Improve answer faithfulness and citation validation.

## License

No license has been specified yet.
