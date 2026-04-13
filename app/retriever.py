from pathlib import Path
import ollama
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "documents"
OLLAMA_MODEL = "llama3.2"
N_RESULTS = 5  # how many chunks to retrieve


def get_chroma_collection():
    """Connect to ChromaDB — same as ingest.py."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    embedding_function = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )


def retrieve_chunks(query: str, n_results: int = N_RESULTS) -> list[str]:
    """Find the most relevant chunks for a query."""
    collection = get_chroma_collection()

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    # results["documents"] is a list of lists — we want the inner list
    return results["documents"][0]


def build_prompt(query: str, chunks: list[str]) -> str:
    """Combine retrieved chunks + question into a prompt."""
    context = "\n\n---\n\n".join(chunks)

    return f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not make anything up.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""


def ask(query: str) -> dict:
    """Full RAG pipeline: query -> retrieve -> generate."""
    print(f"\nQuestion: {query}")

    # Step 1: retrieve relevant chunks
    chunks = retrieve_chunks(query)
    print(f"Retrieved {len(chunks)} chunks")

    # Step 2: build the prompt
    prompt = build_prompt(query, chunks)

    # Step 3: call Ollama
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response["message"]["content"]

    return {
        "question": query,
        "answer": answer,
        "source_chunks": chunks,  # keep these — useful for debugging
    }


if __name__ == "__main__":
    # Quick test
    result = ask("How many days of sick leave do employees get?")
    print(f"\nAnswer: {result['answer']}")
    print("\n--- Source chunks used ---")
    for i, chunk in enumerate(result["source_chunks"]):
        print(f"\nChunk {i+1}:\n{chunk[:200]}")