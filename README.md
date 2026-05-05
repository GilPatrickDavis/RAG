# RAG Pipeline

A minimal Retrieval-Augmented Generation (RAG) system built with Weaviate, Sentence Transformers, and Groq (Llama 3).

Drop `.txt` files into `documents/`, run the pipeline, and ask questions about your documents.

## Stack

- **Weaviate** — vector database (stores and searches document embeddings)
- **Sentence Transformers** — local embedding model (all-MiniLM-L6-v2, runs on your machine)
- **Groq / Llama 3** — free LLM for generating answers

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your-weaviate-key
GROQ_API_KEY=your-groq-key
```

- Weaviate Cloud (free sandbox): https://console.weaviate.cloud
- Groq (free): https://console.groq.com

## Run

```bash
python rag.py
```

On first run it indexes everything in `documents/`. Type `reindex` at the prompt to rebuild after adding new files.
