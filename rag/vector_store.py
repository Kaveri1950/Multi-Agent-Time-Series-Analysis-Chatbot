"""Local, free vector store (FAISS) -- no hosted vector DB, no API key.

Holds three namespaces in one flat index, tagged by `kind`:
  - "news":      embedded headlines/sentiment text  (rag/embed_text.py)
  - "lightcurve": embedded light-curve windows       (astronomy/ts_embedder.py)
  - "report":    embedded past agent-generated reports (for the Explainer
                 agent to retrieve "similar past cases")

FAISS only stores vectors + integer ids; the metadata (kind, source text,
ticker/target, timestamp) is kept alongside in a plain Python list, indexed
by the same id. Good enough for a BTP-scale corpus (hundreds-thousands of
items); swap for a real metadata store only if this ever needs to scale up.
"""

import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine similarity, since embeddings are normalized
        self.metadata: list[dict] = []

    def add(self, vectors: np.ndarray, metadatas: list[dict]) -> None:
        assert len(vectors) == len(metadatas)
        if len(vectors) == 0:
            return
        self.index.add(vectors.astype("float32"))
        self.metadata.extend(metadatas)

    def search(self, query_vector: np.ndarray, k: int = 5, kind: str | None = None) -> list[dict]:
        """Returns up to k nearest metadata entries (optionally filtered to
        one `kind`), each with a 'score' field (cosine similarity, higher=closer)."""
        if self.index.ntotal == 0:
            return []
        query = query_vector.reshape(1, -1).astype("float32")
        # over-fetch if filtering by kind, since FAISS itself doesn't filter
        search_k = min(self.index.ntotal, k * 5 if kind else k)
        scores, ids = self.index.search(query, search_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            if kind is not None and meta.get("kind") != kind:
                continue
            results.append({**meta, "score": float(score)})
            if len(results) >= k:
                break
        return results

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        with open(p / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, default=str)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        p = Path(path)
        index = faiss.read_index(str(p / "index.faiss"))
        store = cls(dim=index.d)
        store.index = index
        with open(p / "metadata.json", "r", encoding="utf-8") as f:
            store.metadata = json.load(f)
        return store


if __name__ == "__main__":
    from embed_text import embed_texts

    news = [
        "Apple stock jumps after strong iPhone sales",
        "Tesla shares fall on delivery miss",
        "Fed holds interest rates steady",
    ]
    vecs = embed_texts(news)

    store = VectorStore(dim=vecs.shape[1])
    store.add(vecs, [{"kind": "news", "text": t} for t in news])

    query_vec = embed_texts(["Apple earnings beat expectations"])[0]
    results = store.search(query_vec, k=2, kind="news")
    for r in results:
        print(f"{r['score']:.3f}  {r['text']}")
