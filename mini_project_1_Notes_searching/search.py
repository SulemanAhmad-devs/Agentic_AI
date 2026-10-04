from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
NOTES_DIR = BASE_DIR / "notes"
CACHE = BASE_DIR / "embeddings_cache.npz"
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def load_notes():
    files = sorted(NOTES_DIR.glob("*.txt"))
    names = [f.name for f in files]
    texts = [f.read_text(encoding="utf-8").strip() for f in files]
    return names, texts


def get_index():
    """Return (names, texts, vectors), reusing the cache if the notes haven't changed."""
    names, texts = load_notes()
    if not texts:
        raise SystemExit(f"No .txt files found in {NOTES_DIR}")
    # ... rest of the function unchanged

    if CACHE.exists():
        cached = np.load(CACHE)
        if cached["texts"].tolist() == texts:
            print("Using cached embeddings.")
            return names, texts, cached["vectors"]

    print("Embedding notes...")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    np.savez(CACHE, vectors=vectors, texts=np.array(texts))
    return names, texts, vectors


def search(query, names, texts, vectors, top_k=3):
    q = model.encode(query, normalize_embeddings=True)
    scores = vectors @ q
    best = np.argsort(-scores)[:top_k]
    return [(names[i], scores[i], texts[i]) for i in best]


def main():
    names, texts, vectors = get_index()
    if not names:
        print(f"No .txt notes found in {NOTES_DIR}. Add a note and try again.")
        return

    print(f"Indexed {len(names)} notes. Type a question (blank line to quit).")

    while True:
        query = input("\n> ").strip()
        if not query:
            break
        for name, score, text in search(query, names, texts, vectors):
            preview = text[:120].replace("\n", " ")
            print(f"  {score:.2f}  [{name}]  {preview}...")


if __name__ == "__main__":
    main()