from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.retriever import ask
from app.ingest import ingest_document

app = FastAPI(
    title="RAG API",
    description="Ask questions about your documents",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    filename: str


@app.get("/")
def root():
    return {"message": "RAG API is running"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = ask(request.question)
    return result


@app.post("/ingest")
def ingest(request: IngestRequest):
    try:
        count = ingest_document(request.filename)
        return {"message": "Document ingested successfully", "chunks_stored": count}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}