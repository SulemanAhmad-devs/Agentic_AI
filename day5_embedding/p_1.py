from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do I reset my password if I forgot it?",
    "Our office is open from 9 AM to 5 PM on weekdays.",
    "You can get a refund within 30 days of purchase.",
    "Shipping usually takes 3 to 5 business days.",
    "Contact support by emailing help@example.com.",
]

# Encode everything in one batch. normalize_embeddings=True gives unit-length vectors.
embeddings = model.encode(sentences, normalize_embeddings=True)
print(embeddings.shape)   # (5, 384): 5 sentences, 384 numbers each
print(embeddings[0][:5])  # first 5 numbers of the first vector

# Cosine similarity between all pairs (a dot product, since vectors are normalized)
similarity = embeddings @ embeddings.T
print(np.round(similarity, 2))