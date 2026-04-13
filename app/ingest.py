from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions


# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"
CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "documents"


def load_pdf(path: Path) -> str:
    """Read all text from a PDF file."""
    reader = PdfReader(path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:  # prevents None errors
            text += page_text + "\n"

    return text


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,   # was 500
        chunk_overlap=100, # was 50
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(text)


def get_chroma_collection():
    """Connect to ChromaDB and return our collection."""

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    # Free local embedding model
    embedding_function = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    return collection


def ingest_document(pdf_filename: str):
    """Full pipeline: PDF -> chunks -> embeddings -> ChromaDB."""

    pdf_path = DATA_PATH / pdf_filename

    print(f"Loading {pdf_path}...")

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found at: {pdf_path}"
        )

    text = load_pdf(pdf_path)

    print(f"Extracted {len(text)} characters")

    print("Chunking text...")
    chunks = chunk_text(text)

    print(f"Created {len(chunks)} chunks")

    print("Storing in ChromaDB...")

    collection = get_chroma_collection()

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.upsert(
        documents=chunks,
        ids=ids,
    )

    print(f"Done. {len(chunks)} chunks stored in ChromaDB.")

    return len(chunks)


if __name__ == "__main__":
    ingest_document("document.pdf")
