"""Measure model-selection accuracy across the three strategies.

Produces the table the thesis needs: does constraining the LLM to feasible
models actually help, or would either half alone do as well?

  filter_only : deterministic. Classify the task, keep what can physically
                answer, take the best measured skill. No LLM call.
  llm_only    : the whole registry handed to the model, free choice.
  hybrid      : the filter narrows, the LLM ranks, a validator rejects
                anything off the shortlist.

Reported per category as well as overall, because the categories test
different things and averaging them hides which part is failing. Adversarial
cases -- where the naive reading is wrong -- are the ones worth watching: a
strategy can score well overall by handling plain phrasings and still be
useless on the cases a real user actually gets wrong.

Results are written to JSON so the report can cite them without re-running,
and so a later run can be compared against an earlier one.
"""

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from agents.model_selector import select_model
from evaluation.query_set import QUERIES, stats

RESULTS_DIR = Path(__file__).parent / "results"


def run_strategy(strategy: str, queries=None, verbose: bool = True,
                 backend: str | None = None) -> dict:
    """Score one strategy over the labelled set."""
    queries = queries or QUERIES
    rows, started = [], time.time()

    for i, case in enumerate(queries, 1):
        t0 = time.time()
        try:
            sel = select_model(case["query"], domain=case["domain"],
                               available=case["available"], strategy=strategy,
                               backend=backend)
            chosen = sel["model_id"]
            fallback = sel["fallback_used"]
            task = sel["task"]
        except Exception as e:
            chosen, fallback, task = f"ERROR:{type(e).__name__}", True, None

        rows.append({
            "query": case["query"], "domain": case["domain"],
            "category": case["category"], "expected": case["expected"],
            "chosen": chosen, "correct": chosen == case["expected"],
            "task": task, "fallback_used": fallback,
            "seconds": round(time.time() - t0, 2),
        })
        if verbose and i % 25 == 0:
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"  [{i}/{len(queries)}] running accuracy {acc:.1%}", flush=True)

    return {
        "strategy": strategy,
        "n": len(rows),
        "elapsed_seconds": round(time.time() - started, 1),
        "accuracy": round(sum(r["correct"] for r in rows) / max(len(rows), 1), 4),
        "by_category": _breakdown(rows, "category"),
        "by_domain": _breakdown(rows, "domain"),
        "fallback_rate": round(sum(r["fallback_used"] for r in rows) / max(len(rows), 1), 4),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / max(len(rows), 1), 2),
        "rows": rows,
    }


def _breakdown(rows, key) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r["correct"])
    return {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)}
            for k, v in sorted(groups.items())}


def confusions(result: dict, limit: int = 12) -> list:
    """The mistakes, most frequent first -- more informative than the score.

    A strategy that confuses forecast with evaluation is failing at the task
    taxonomy; one that picks a plausible model from the wrong domain is
    failing at routing. The aggregate accuracy cannot tell those apart.
    """
    wrong = [(r["expected"], r["chosen"], r["query"]) for r in result["rows"] if not r["correct"]]
    counts = Counter((e, c) for e, c, _ in wrong)
    out = []
    for (expected, chosen), n in counts.most_common(limit):
        example = next(q for e, c, q in wrong if e == expected and c == chosen)
        out.append({"expected": str(expected), "chosen": str(chosen),
                    "count": n, "example": example})
    return out


def compare(results: list) -> str:
    """The headline table."""
    cats = sorted({c for r in results for c in r["by_category"]})
    header = f"{'strategy':<14}{'overall':>9}" + "".join(f"{c[:11]:>13}" for c in cats) \
             + f"{'mean s':>9}"
    lines = [header, "-" * len(header)]
    for r in results:
        row = f"{r['strategy']:<14}{r['accuracy']:>8.1%}"
        for c in cats:
            entry = r["by_category"].get(c)
            row += f"{entry['accuracy']:>12.1%} " if entry else f"{'--':>13}"
        row += f"{r['mean_seconds']:>9.2f}"
        lines.append(row)
    return "\n".join(lines)


def save(results: list, name: str = "selection") -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps({
        "query_set": stats(),
        "results": [{k: v for k, v in r.items() if k != "rows"} for r in results],
        "confusions": {r["strategy"]: confusions(r) for r in results},
        "rows": {r["strategy"]: r["rows"] for r in results},
    }, indent=2))
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Measure model-selection accuracy.")
    ap.add_argument("--strategies", default="filter_only",
                    help="comma-separated: filter_only,llm_only,hybrid")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--name", default="selection")
    args = ap.parse_args()

    queries = QUERIES[:args.limit] if args.limit else QUERIES
    print(f"Query set: {len(queries)} queries\n" + json.dumps(
        {k: v for k, v in stats().items() if k != "by_expected"}, indent=2))

    results = []
    for strategy in args.strategies.split(","):
        strategy = strategy.strip()
        print(f"\n=== {strategy} ===", flush=True)
        results.append(run_strategy(strategy, queries))

    print("\n" + compare(results))

    for r in results:
        print(f"\n--- {r['strategy']}: most common mistakes ---")
        for c in confusions(r, limit=8):
            print(f"  {c['count']:>3}x  expected {c['expected']:<32} got {c['chosen']}")
            print(f"        e.g. {c['example']!r}")

    print(f"\nsaved -> {save(results, args.name)}")
