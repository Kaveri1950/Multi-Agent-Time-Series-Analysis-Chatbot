"""Free, local text embeddings -- no OpenAI/Cohere embedding API involved.

Uses sentence-transformers' all-MiniLM-L6-v2: pretrained, open-source
(Apache-2.0), ~90MB, runs fast on CPU (~a few ms/sentence). This answers
the "do I need a paid embeddings API" question for the text side of the
project -- no, this is free and local.
"""

import numpy as np

_model = None
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns an (N, 384) float32 array of embeddings."""
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")


if __name__ == "__main__":
    sample = [
        "Apple stock jumps after strong iPhone sales in Q3",
        "A transit-like dip was detected in the light curve of KIC 11446443",
        "Fed signals possible rate cut, markets rally",
    ]
    vecs = embed_texts(sample)
    print("Embedding shape:", vecs.shape)
    sims = vecs @ vecs.T
    print("Cosine similarity matrix:\n", sims.round(3))
