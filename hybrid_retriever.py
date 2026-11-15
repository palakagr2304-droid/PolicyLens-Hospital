import json
import os
import joblib
import requests
import numpy as np

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = BASE_DIR / "output" / "chunks.json"

TFIDF_MATRIX_FILE = (
    BASE_DIR / "output" / "tfidf_matrix.pkl"
)

TFIDF_VECTORIZER_FILE = (
    BASE_DIR / "output" / "tfidf_vectorizer.pkl"
)

EMBEDDINGS_FILE = (
    BASE_DIR / "output" / "embeddings.npy"
)


# ---------------------------------------------------------
# Load chunks
# ---------------------------------------------------------

with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


print(f"Loaded {len(chunks)} chunks.")


# ---------------------------------------------------------
# Load TF-IDF
# ---------------------------------------------------------

vectorizer = joblib.load(
    TFIDF_VECTORIZER_FILE
)

tfidf_matrix = joblib.load(
    TFIDF_MATRIX_FILE
)


# ---------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------

embeddings = np.load(
    EMBEDDINGS_FILE
)


print(
    f"TF-IDF matrix shape: {tfidf_matrix.shape}"
)

print(
    f"Embedding matrix shape: {embeddings.shape}"
)


# ---------------------------------------------------------
# Embedding model management (Lazy loaded & Cloud-optimized)
# ---------------------------------------------------------

_model = None


def get_question_embedding(question: str) -> np.ndarray:
    """
    Generate normalized embedding for user query.
    1. If HF_TOKEN is configured in environment, queries Hugging Face Serverless
       Inference API (0 MB local RAM footprint — ideal for Render 512MB free tier).
    2. Otherwise, lazy-loads local SentenceTransformer model with strict single-thread
       CPU memory optimizations.
    """
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        headers = {
            "Authorization": f"Bearer {hf_token.strip()}",
            "Content-Type": "application/json",
        }
        urls = [
            "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction",
            "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2",
        ]
        payloads = [
            {"inputs": [question]},
            {"inputs": question},
        ]
        for api_url in urls:
            for payload in payloads:
                try:
                    res = requests.post(api_url, headers=headers, json=payload, timeout=8)
                    if res.status_code == 200:
                        data = res.json()
                        emb = np.array(data, dtype=np.float32)
                        if emb.ndim == 1 and emb.shape[0] == 384:
                            emb = np.expand_dims(emb, axis=0)
                        if emb.ndim == 2 and emb.shape[1] == 384:
                            norm = np.linalg.norm(emb, axis=1, keepdims=True)
                            return emb / np.maximum(norm, 1e-12)
                except Exception as e:
                    print(f"[Embedding API Warning] {api_url} failed: {e}")

    # If running on Render (512MB free tier), avoid loading local PyTorch to prevent OOM kills
    if os.getenv("RENDER"):
        print("[Embedding Info] Running in Render cloud environment; using TF-IDF ranking fallback.")
        return None

    # Local fallback: lazy load model on first query (for local machines)
    global _model
    if _model is None:
        print("\nLoading local embedding model...")
        import torch
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)
        from sentence_transformers import SentenceTransformer
        try:
            _model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                local_files_only=True,
                device="cpu"
            )
        except Exception:
            _model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu"
            )
        _model.eval()
        print("Local embedding model loaded.")

    return _model.encode([question], normalize_embeddings=True)


# ---------------------------------------------------------
# Retrieval function
# ---------------------------------------------------------

def retrieve(
    question,
    top_k=5,
    tfidf_weight=0.4,
    embedding_weight=0.6
):

    # -----------------------------------------------------
    # Create TF-IDF vector for question
    # -----------------------------------------------------

    question_tfidf = vectorizer.transform(
        [question]
    )


    # -----------------------------------------------------
    # Calculate TF-IDF similarity
    # -----------------------------------------------------

    tfidf_scores = cosine_similarity(
        question_tfidf,
        tfidf_matrix
    )[0]


    # -----------------------------------------------------
    # Create embedding for question
    # -----------------------------------------------------

    question_embedding = get_question_embedding(question)


    # -----------------------------------------------------
    # Calculate embedding similarity & hybrid score
    # -----------------------------------------------------

    if question_embedding is not None:
        embedding_scores = cosine_similarity(
            question_embedding,
            embeddings
        )[0]
        hybrid_scores = (
            tfidf_weight * tfidf_scores
            +
            embedding_weight * embedding_scores
        )
    else:
        embedding_scores = np.zeros_like(tfidf_scores)
        hybrid_scores = tfidf_scores


    # -----------------------------------------------------
    # Get top results
    # -----------------------------------------------------

    top_indices = (
        np.argsort(hybrid_scores)[-top_k:][::-1]
    )


    # -----------------------------------------------------
    # Build results
    # -----------------------------------------------------

    results = []

    for index in top_indices:

        chunk = chunks[index]

        result = {
            "chunk_id": chunk["chunk_id"],
            "document": chunk["document"],
            "policy_id": chunk["policy_id"],
            "section": chunk["section"],
            "pages": chunk["pages"],
            "text": chunk["text"],
            "tfidf_score": float(
                tfidf_scores[index]
            ),
            "embedding_score": float(
                embedding_scores[index]
            ),
            "hybrid_score": float(
                hybrid_scores[index]
            )
        }

        results.append(result)


    return results


# ---------------------------------------------------------
# Interactive testing
# ---------------------------------------------------------

if __name__ == "__main__":

    while True:

        question = input(
            "\nAsk a hospital policy question "
            "(or type 'exit'): "
        )


        if question.lower() == "exit":
            break


        results = retrieve(
            question,
            top_k=5
        )


        print("\n")
        print("=" * 75)
        print("TOP HYBRID RETRIEVAL RESULTS")
        print("=" * 75)


        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nRESULT {rank}"
            )

            print("-" * 75)

            print(
                f"Hybrid Score: "
                f"{result['hybrid_score']:.4f}"
            )

            print(
                f"TF-IDF Score: "
                f"{result['tfidf_score']:.4f}"
            )

            print(
                f"Embedding Score: "
                f"{result['embedding_score']:.4f}"
            )

            print(
                f"Document: "
                f"{result['document']}"
            )

            print(
                f"Policy ID: "
                f"{result['policy_id']}"
            )

            print(
                f"Section: "
                f"{result['section']}"
            )

            print(
                f"Pages: "
                f"{result['pages']}"
            )

            print(
                f"\n{result['text']}"
            )