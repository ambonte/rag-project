# RAG API

A production-style Retrieval-Augmented Generation (RAG) system built with FastAPI, ChromaDB, and Groq. Upload any PDF and query it with natural language.

## What it does

Plain LLMs can only answer from their training data. This system lets you give the LLM your own documents — company policies, research papers, manuals — and ask questions about them. The LLM answers using only your content, not guesswork.

## Architecture
PDF → text extraction → chunking → embeddings → ChromaDB
↓
User question → embedding → vector search → top 5 chunks → Groq LLM → answer

## Tech stack

- **FastAPI** — REST API
- **ChromaDB** — local vector database
- **all-MiniLM-L6-v2** — open-source embedding model (runs locally, no API cost)
- **Groq + Llama 3** — LLM inference (free tier)
- **Railway** — deployment

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Status check |
| POST | `/ingest` | Load a PDF into the vector DB |
| POST | `/ask` | Ask a question, get an answer with source chunks |

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/rag-project
cd rag-project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Ingest a document:
```bash
python -m app.ingest
```

Start the API:
```bash
uvicorn app.main:app --reload
```

## Example usage

```bash
curl -X POST https://your-railway-url/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the parental leave policy?"}'
```

Response:
```json
{
  "question": "What is the parental leave policy?",
  "answer": "Primary caregivers are entitled to 16 weeks of fully paid parental leave...",
  "source_chunks": ["...relevant document excerpts..."]
}
```

## Evaluation

The system was evaluated against 8 ground-truth Q&A pairs derived from the test document, scoring **89% overall (8/8 questions passed)**.

Scoring uses keyword matching — a deliberately simple baseline. Its known limitation: the LLM sometimes paraphrases a correct answer without repeating the question's exact words, which causes false negatives (e.g. answering "10 days per year" for a sick leave question scores lower than expected because it doesn't repeat "sick leave" or "paid"). 

In production, this would be replaced with an LLM-as-judge approach: a second LLM call asking "given this context, is this answer correct?" which is more robust to paraphrasing.

## Key engineering decisions

**Chunk size: 800 characters with 100 overlap**
Started at 500/50 based on common defaults. Evaluation revealed chunks were splitting mid-section, causing retrieval to return incomplete policy paragraphs. Increasing to 800/100 kept full policy sections intact and improved retrieval quality.

**Embedding model: all-MiniLM-L6-v2**
Chosen for being free, fast, and well-benchmarked on semantic similarity tasks. Tradeoff vs OpenAI embeddings: slightly lower accuracy on nuanced queries, but zero API cost and no latency from external calls.

**N_RESULTS: 5 chunks retrieved**
Started at 3. Increasing to 5 improved recall on questions where the answer spanned multiple document sections, at the cost of a slightly longer prompt.

## What I'd improve with more time

- Replace keyword eval with LLM-as-judge scoring
- Add hybrid search (dense embeddings + BM25 keyword search) for better recall
- Persist ChromaDB to a hosted vector DB (Pinecone/Weaviate) instead of committing to the repo
- Add a reranking step (Cohere rerank API) to improve chunk ordering
- Stream the LLM response instead of waiting for the full answer
