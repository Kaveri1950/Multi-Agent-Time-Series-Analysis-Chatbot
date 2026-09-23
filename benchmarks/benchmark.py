"""System performance benchmark -- TECHNICAL_PLAN.md Section 7 /
PROFESSOR_PROPOSAL.md Section 5: run this unchanged on the laptop now, and
again on the lab parallel-computing system once available, then diff the
two JSON outputs for the report's "System Feasibility" section.

Measures the three genuinely slow parts of the pipeline:
  1. Local LLM (Ollama) latency -- the thing the original proposal flagged
     as the open question.
  2. Text embedding throughput (MiniLM).
  3. Light-curve autoencoder training time.

Usage:
    python benchmarks/benchmark.py --tag laptop-cpu
    python benchmarks/benchmark.py --tag lab-gpu
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np


def bench_llm(model: str = "llama3.2:3b", n_calls: int = 3) -> dict:
    from agents.llm_backend import call_llm

    prompt = "In two sentences, explain what an anomaly detector does."
    call_llm("hi", backend="ollama", model=model)  # warmup: Ollama loads the model into RAM on first request

    latencies = []
    for _ in range(n_calls):
        start = time.perf_counter()
        call_llm(prompt, backend="ollama", model=model)
        latencies.append(time.perf_counter() - start)
    return {
        "model": model,
        "n_calls": n_calls,
        "mean_latency_sec": round(float(np.mean(latencies)), 2),
        "min_latency_sec": round(float(np.min(latencies)), 2),
        "max_latency_sec": round(float(np.max(latencies)), 2),
    }


def bench_embeddings(n_texts: int = 100) -> dict:
    from rag.embed_text import embed_texts

    embed_texts(["warmup"])  # trigger one-time model load/JIT outside the timed region

    texts = [f"Sample financial news headline number {i} about markets and earnings." for i in range(n_texts)]
    start = time.perf_counter()
    embed_texts(texts)
    elapsed = time.perf_counter() - start
    return {
        "n_texts": n_texts,
        "total_sec": round(elapsed, 3),
        "texts_per_sec": round(n_texts / elapsed, 1),
    }


def bench_autoencoder_training(n_windows: int = 200, window_size: int = 200, epochs: int = 30) -> dict:
    from astronomy.ts_embedder import train_autoencoder

    rng = np.random.default_rng(0)
    windows = rng.normal(1.0, 0.01, size=(n_windows, window_size)).astype("float32")

    start = time.perf_counter()
    train_autoencoder(windows, epochs=epochs)
    elapsed = time.perf_counter() - start
    return {
        "n_windows": n_windows,
        "epochs": epochs,
        "total_sec": round(elapsed, 2),
        "sec_per_epoch": round(elapsed / epochs, 3),
    }


def run_all(tag: str, llm_model: str, skip_llm: bool) -> dict:
    print(f"[benchmark] running on: {platform.processor() or platform.machine()}, {platform.system()} {platform.release()}")

    results = {"tag": tag, "platform": platform.platform(), "processor": platform.processor()}

    print("[benchmark] embeddings...")
    results["embeddings"] = bench_embeddings()
    print(" ", results["embeddings"])

    print("[benchmark] autoencoder training...")
    results["autoencoder"] = bench_autoencoder_training()
    print(" ", results["autoencoder"])

    if not skip_llm:
        print(f"[benchmark] local LLM ({llm_model}) -- this is the slow one, be patient...")
        results["llm"] = bench_llm(model=llm_model)
        print(" ", results["llm"])
    else:
        print("[benchmark] skipping LLM benchmark (--skip-llm)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="laptop-cpu", help="Label for this run, e.g. 'laptop-cpu' or 'lab-gpu'")
    parser.add_argument("--llm-model", default="llama3.2:3b")
    parser.add_argument("--skip-llm", action="store_true", help="Skip the (slow) LLM latency benchmark")
    args = parser.parse_args()

    results = run_all(args.tag, args.llm_model, args.skip_llm)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.tag}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")
