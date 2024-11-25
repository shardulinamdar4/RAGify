from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def evaluate_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    """
    Evaluates chunks based on semantic coherence and token count.
    """
    model = SentenceTransformer(model_name)

    # Compute embeddings for all chunks
    embeddings = [model.encode(chunk) for chunk in chunks]

    # Compute pairwise similarities
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        similarities.append(sim)

    avg_similarity = sum(similarities) / len(similarities) if similarities else 0

    return {
        "avg_similarity": avg_similarity,
        "chunk_count": len(chunks)
    }