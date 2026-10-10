from functools import lru_cache
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = BASE_DIR / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

COLLECTION_NAME = "pdf_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"


# -----------------------------
# Load the embedding model
# -----------------------------

@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load the model once per Django process.
    The first load may download the model.
    """
    return SentenceTransformer(MODEL_NAME)


# -----------------------------
# Connect to ChromaDB
# -----------------------------

@lru_cache(maxsize=1)
def get_collection():
    """
    Open or create a persistent Chroma collection.
    Data remains on disk between server restarts.
    """
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# -----------------------------
# Extract PDF text
# -----------------------------

def extract_pdf_pages(pdf_path):
    """
    Return a list of dictionaries.
    Each dictionary contains a page number and its text.
    """
    reader = PdfReader(str(pdf_path))

    if reader.is_encrypted:
        raise ValueError(
            "This PDF is password-protected. "
            "Please upload an unlocked PDF."
        )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip(),
            })

    if not pages:
        raise ValueError(
            "No readable text was found. "
            "The PDF may be scanned or image-only."
        )

    return pages


# -----------------------------
# Chunk text with overlap
# -----------------------------

def chunk_text(text, chunk_size=CHUNK_SIZE,
               overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping character-based chunks.

    Example:
    Chunk 1: characters 0-999
    Chunk 2: characters 800-1799

    Therefore, 200 characters overlap.
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative "
            "and smaller than chunk size."
        )

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


# -----------------------------
# Ingest PDF into ChromaDB
# -----------------------------

def ingest_pdf(pdf_path, document_id, filename):
    """
    Extract, chunk, embed, and store a PDF.

    document_id uniquely identifies this upload.
    Metadata lets us retrieve the source page later.
    """
    pages = extract_pdf_pages(pdf_path)

    all_chunks = []
    all_metadata = []

    for page_data in pages:
        page_chunks = chunk_text(page_data["text"])

        for chunk_number, chunk in enumerate(
            page_chunks,
            start=1,
        ):
            all_chunks.append(chunk)

            all_metadata.append({
                "document_id": document_id,
                "source": filename,
                "page": page_data["page"],
                "chunk_number": chunk_number,
            })

    if not all_chunks:
        raise ValueError(
            "No usable text chunks could be created."
        )

    model = get_embedding_model()

    embeddings = model.encode(
        all_chunks,
        normalize_embeddings=True,
    ).tolist()

    collection = get_collection()

    # Unique IDs prevent chunks from different uploads
    # from accidentally overwriting each other.
    chunk_ids = [
        f"{document_id}_{index}"
        for index in range(len(all_chunks))
    ]

    # ChromaDB's add operation accepts batches.
    batch_size = 100

    for start in range(0, len(all_chunks), batch_size):
        end = start + batch_size

        collection.add(
            ids=chunk_ids[start:end],
            documents=all_chunks[start:end],
            embeddings=embeddings[start:end],
            metadatas=all_metadata[start:end],
        )

    return {
        "pages": len(pages),
        "chunks": len(all_chunks),
    }


# -----------------------------
# Semantic search
# -----------------------------

def search_pdf(query, document_id, top_k=5):
    """
    Return the most relevant chunks from one uploaded PDF.

    ChromaDB cosine distance:
    Lower distance = more similar.

    Similarity = 1 - cosine distance.
    """
    query = query.strip()

    if not query:
        return []

    collection = get_collection()

    # Do not query an empty collection.
    if collection.count() == 0:
        return []

    # Only retrieve chunks belonging to this upload.
    document_count = collection.get(
        where={"document_id": document_id},
        include=[],
    )["ids"]

    if document_count == 0:
        return []
    document_count = len(document_count)



    top_k = max(1, min(top_k, document_count))

    model = get_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={"document_id": document_id},
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    matches = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        similarity = 1.0 - distance

        matches.append({
            "text": document,
            "source": metadata["source"],
            "page": metadata["page"],
            "chunk_number": metadata["chunk_number"],
            "score": round(similarity, 4),
            "distance": round(distance, 4),
        })

    return matches