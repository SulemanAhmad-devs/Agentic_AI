import chromadb
from sentence_transformers import SentenceTransformer

from data.course_notes import COURSE_NOTES


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Create persistent Chroma database
client = chromadb.PersistentClient(
    path="./chroma_db"
)


# Create or get collection
collection = client.get_or_create_collection(
    name="course_notes"
)


def setup_database():
    """
    Add course notes and their embeddings to Chroma.
    """

    documents = [
        note["text"]
        for note in COURSE_NOTES
    ]

    ids = [
        note["id"]
        for note in COURSE_NOTES
    ]

    metadatas = [
        {
            "category": note["category"]
        }
        for note in COURSE_NOTES
    ]

    # Generate embeddings
    embeddings = model.encode(documents).tolist()

    # Store in Chroma
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("Course notes added to Chroma.")


def search_course(query, category=None):
    """
    Search Chroma for relevant course information.
    """

    # Convert question into embedding
    query_embedding = model.encode(query).tolist()

    # Search
    if category:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            where={"category": category},
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
        )

    return results