import os
from pathlib import Path
from dotenv import load_dotenv
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.data import DataObject
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

COLLECTION_NAME = "Documents"
DOCUMENTS_FOLDER = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 500
OVERLAP = 100
TOP_K = 3


def load_documents(folder: str) -> list[dict]:
    return [
        {"filename": p.name, "content": p.read_text(encoding="utf-8")}
        for p in Path(folder).glob("*.txt")
    ]


def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE]
        if len(chunk) < CHUNK_SIZE * 0.1 and chunks:
            chunks[-1] += chunk
            break
        chunks.append(chunk)
        start += CHUNK_SIZE - OVERLAP
    return chunks


def create_chunks(documents: list[dict]) -> list[dict]:
    return [
        {"source": doc["filename"], "chunk_index": i, "content": text}
        for doc in documents
        for i, text in enumerate(chunk_text(doc["content"]))
    ]


def embed_chunks(model: SentenceTransformer, chunks: list[dict]) -> list[dict]:
    embeddings = model.encode([c["content"] for c in chunks], normalize_embeddings=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def connect_to_weaviate() -> weaviate.WeaviateClient:
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=weaviate.auth.AuthApiKey(os.environ["WEAVIATE_API_KEY"]),
    )


def build_collection(client: weaviate.WeaviateClient):
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
    return client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
        ],
    )


def store_chunks(collection, chunks: list[dict]):
    collection.data.insert_many([
        DataObject(
            properties={"content": c["content"], "source": c["source"], "chunk_index": c["chunk_index"]},
            vector=c["embedding"],
        )
        for c in chunks
    ])


def retrieve(query: str, model: SentenceTransformer, client: weaviate.WeaviateClient) -> list[dict]:
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    results = client.collections.get(COLLECTION_NAME).query.near_vector(
        near_vector=query_vector,
        limit=TOP_K,
        return_properties=["content", "source", "chunk_index"],
        return_metadata=MetadataQuery(distance=True),
    )
    return [
        {
            "content": o.properties["content"],
            "source": o.properties["source"],
            "chunk_index": o.properties["chunk_index"],
            "similarity": round(1 - o.metadata.distance, 4),
        }
        for o in results.objects
    ]


def generate_response(query: str, chunks: list[dict]) -> str:
    system_prompt = (
        "You are a helpful assistant that answers questions based strictly on the provided context. "
        "Answer only using the context. If the answer is not there, say so. Cite the source document when relevant."
    )

    context_block = "".join(
        f"\n--- CONTEXT {i} (from: {c['source']}) ---\n{c['content']}\n"
        for i, c in enumerate(chunks, 1)
    )

    response = Groq(api_key=os.environ["GROQ_API_KEY"]).chat.completions.create(
        model=LLM_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
        ],
    )
    return response.choices[0].message.content


def index_documents(model: SentenceTransformer, client: weaviate.WeaviateClient):
    print("Indexing documents...")
    chunks = create_chunks(load_documents(DOCUMENTS_FOLDER))
    chunks = embed_chunks(model, chunks)
    store_chunks(build_collection(client), chunks)
    print(f"Done — {len(chunks)} chunks indexed.\n")


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to Weaviate...")
    client = connect_to_weaviate()

    try:
        if not client.collections.exists(COLLECTION_NAME):
            index_documents(model, client)
        else:
            count = client.collections.get(COLLECTION_NAME).aggregate.over_all(total_count=True).total_count
            print(f"Using existing index ({count} chunks). Type 'reindex' to rebuild.\n")

        print("Ready. Type 'quit' to exit.\n")

        while True:
            query = input("You: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit"):
                break
            if query.lower() == "reindex":
                index_documents(model, client)
                continue

            chunks = retrieve(query, model, client)
            if not chunks:
                print("No relevant documents found.\n")
                continue

            sources = ", ".join(f"{c['source']} ({c['similarity']})" for c in chunks)
            print(f"\n[Sources: {sources}]")
            print(f"\nAnswer: {generate_response(query, chunks)}\n")
            print("-" * 60)

    finally:
        client.close()


if __name__ == "__main__":
    main()
