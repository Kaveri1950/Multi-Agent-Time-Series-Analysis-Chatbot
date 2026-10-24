"""Search the NSE corporate filings by meaning, not just by keyword.

The 60,905 cached announcements for the 50 panel stocks are deduplicated on
(company, text) and embedded with the same local MiniLM model the RAG layer
uses. A query like "an independent director stepped down" finds filings that
say "cessation of ... as Independent Director", which a keyword search misses.

Three retrievers are available so they can be compared honestly:
  keyword  TF-IDF over unigrams and bigrams
  dense    MiniLM cosine similarity (FAISS inner product on unit vectors)
  hybrid   reciprocal-rank fusion of the two

Build the index:  python -m finance.filing_search --build      (a few minutes on CPU)
Search:           python -m finance.filing_search "credit rating downgrade" --symbol TATASTEEL
"""

from __future__ import annotations

import argparse
import glob
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
SOURCE = ROOT / "data" / "cache" / "news" / "nse_announcements"
INDEX_DIR = ROOT / "data" / "cache" / "filing_index"
RRF_K = 60


def load_filings() -> pd.DataFrame:
    frames = [pd.read_parquet(f).assign(symbol=Path(f).stem) for f in glob.glob(str(SOURCE / "*.parquet"))]
    if not frames:
        raise FileNotFoundError(f"no filings under {SOURCE}; run finance.news_history first")
    d = pd.concat(frames, ignore_index=True)
    d["text"] = d["text"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    d = d[d["text"].str.len() > 0]
    raw = len(d)
    d = d.drop_duplicates(["company", "text"]).sort_values("timestamp").reset_index(drop=True)
    d.attrs["raw_rows"] = raw
    return d


def build(batch: int = 256) -> None:
    from rag.embed_text import embed_texts

    d = load_filings()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vecs = np.vstack([embed_texts(d["text"].iloc[i:i + batch].tolist())
                      for i in range(0, len(d), batch)])
    np.save(INDEX_DIR / "embeddings.npy", vecs.astype("float32"))
    d.to_parquet(INDEX_DIR / "filings.parquet", index=False)
    print(f"indexed {len(d):,} unique filings (from {d.attrs['raw_rows']:,} rows) -> {INDEX_DIR}")


@lru_cache(maxsize=1)
def _index():
    import faiss
    from sklearn.feature_extraction.text import TfidfVectorizer

    d = pd.read_parquet(INDEX_DIR / "filings.parquet")
    vecs = np.load(INDEX_DIR / "embeddings.npy")
    dense = faiss.IndexFlatIP(vecs.shape[1])
    dense.add(vecs)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
    sparse = tfidf.fit_transform(d["text"])
    return d, dense, tfidf, sparse


def _rank(query: str, method: str, pool: int) -> list[int]:
    d, dense, tfidf, sparse = _index()
    if method == "keyword":
        s = (sparse @ tfidf.transform([query]).T).toarray().ravel()
        return list(np.argsort(-s)[:pool])
    if method == "dense":
        from rag.embed_text import embed_texts
        _, ids = dense.search(embed_texts([query]), pool)
        return [int(i) for i in ids[0] if i >= 0]
    if method == "hybrid":
        score: dict[int, float] = {}
        for m in ("keyword", "dense"):
            for r, i in enumerate(_rank(query, m, pool)):
                score[i] = score.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
        return sorted(score, key=lambda i: -score[i])
    raise ValueError(f"unknown method {method!r}")


def search(query: str, k: int = 10, method: str = "hybrid", symbol: str | None = None) -> pd.DataFrame:
    """Top-k filings for `query`, optionally limited to one stock."""
    d = _index()[0]
    pool = 2000 if symbol else max(200, k * 5)
    ids = _rank(query, method, pool)
    if symbol:
        sym = symbol.upper().replace(".NS", "")
        ids = [i for i in ids if d.at[i, "symbol"] == sym]
    out = d.iloc[ids[:k]][["timestamp", "symbol", "type", "text"]].copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--symbol")
    ap.add_argument("--method", default="hybrid", choices=["keyword", "dense", "hybrid"])
    ap.add_argument("-k", type=int, default=10)
    a = ap.parse_args()
    if a.build:
        build()
    if a.query:
        with pd.option_context("display.max_colwidth", 120, "display.width", 200):
            print(search(a.query, a.k, a.method, a.symbol))


if __name__ == "__main__":
    main()
