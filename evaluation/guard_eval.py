"""Does the number guard catch invented figures, and what does it miss?

The guard (agents/answer_style.unsupported_numbers) flags any figure in an
LLM answer that is not within 5% of a number in the computed facts. It has
been used for months without its own test. Here it gets one.

SET-UP. Real fact blocks from the portfolio-risk module (three model
portfolios). Each fact line is turned into an "answer" that restates it, so
the starting answer is correct by construction. Then exactly one number in
it is corrupted, in one of several realistic ways, and we record whether the
guard flags the corrupted figure.

  CORRECT ANSWERS (false-positive test)
    verbatim        the line as written
    rounded         every number rounded to whole units (what a model does)
  ERRORS THE GUARD SHOULD CATCH
    fabricated      a random figure of similar size, 0.2x to 5x
    off_by_20pct    the right figure x 1.2
    off_by_10pct    the right figure x 1.1
    decimal_shift   the right figure x 10
    digit_swap      first two digits swapped (12.09 -> 21.09)
  KNOWN BLIND SPOTS (reported, not hidden)
    small_3pct      x 1.03 -- inside the 5% tolerance by design
    sign_flip       -1.21% written as 1.21% -- the guard compares magnitudes
    swapped_fact    a different number from the same facts -- it exists, so it passes

Run:  python -m evaluation.guard_eval       (about a minute)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT_PATH = Path(__file__).parent / "results" / "guard_eval.json"
NUM = re.compile(r"-?\d+(?:\.\d+)?")
SEED = 0
CATCH = ("fabricated", "off_by_20pct", "off_by_10pct", "decimal_shift", "digit_swap")
BLIND = ("small_3pct", "sign_flip", "swapped_fact")


def _fmt(value: float, like: str) -> str:
    dec = len(like.split(".")[1]) if "." in like else 0
    return f"{value:.{dec}f}"


def corrupt(token: str, kind: str, facts_numbers: list[str], rng) -> str | None:
    v = float(token)
    if kind == "fabricated":
        return _fmt(v * rng.uniform(0.2, 5.0), token)
    if kind == "off_by_20pct":
        return _fmt(v * 1.2, token)
    if kind == "off_by_10pct":
        return _fmt(v * 1.1, token)
    if kind == "decimal_shift":
        return _fmt(v * 10, token)
    if kind == "small_3pct":
        return _fmt(v * 1.03, token)
    if kind == "digit_swap":
        sign, body = ("-", token[1:]) if token.startswith("-") else ("", token)
        digits = [c for c in body if c.isdigit()]
        if len(digits) < 2 or digits[0] == digits[1] or digits[1] == "0" and "." not in body[:2]:
            return None
        i, j = [k for k, c in enumerate(body) if c.isdigit()][:2]
        b = list(body)
        b[i], b[j] = b[j], b[i]
        return sign + "".join(b)
    if kind == "sign_flip":
        return token[1:] if token.startswith("-") else None
    if kind == "swapped_fact":
        others = [n for n in facts_numbers if abs(abs(float(n)) - abs(v)) > 0.05 * max(abs(v), 1e-9)]
        return str(rng.choice(others)) if others else None
    raise ValueError(kind)


def main():
    from agents.answer_style import _INNOCUOUS, unsupported_numbers
    from quant.portfolio_risk import MODEL_PORTFOLIOS, analyse, facts_for_llm

    rng = np.random.default_rng(SEED)
    # Industry lines were added to the fact block after this evaluation was run; they are
    # left out so the experiment (and evaluation/results/guard_eval.json) stays reproducible.
    blocks = {name: "\n".join(line for line in facts_for_llm(analyse(h, with_backtest=False)).splitlines()
                              if not line.startswith("- industry"))
              for name, h in MODEL_PORTFOLIOS.items()}

    trials = {k: {"n": 0, "flagged": 0} for k in ("verbatim", "rounded") + CATCH + BLIND}
    misses = {k: [] for k in CATCH}
    for name, facts in blocks.items():
        facts_numbers = [m.group(0) for m in NUM.finditer(facts)]
        for line in facts.splitlines()[1:]:
            answer = line.lstrip("- ").strip()
            spots = [m for m in NUM.finditer(answer)
                     if m.group(0).lstrip("-") not in _INNOCUOUS and abs(float(m.group(0))) >= 0.1]
            if not spots:
                continue
            trials["verbatim"]["n"] += 1
            trials["verbatim"]["flagged"] += bool(unsupported_numbers(answer, facts))
            rounded = NUM.sub(lambda m: str(round(float(m.group(0)))), answer)
            trials["rounded"]["n"] += 1
            trials["rounded"]["flagged"] += bool(unsupported_numbers(rounded, facts))
            for m in spots:
                for kind in CATCH + BLIND:
                    new = corrupt(m.group(0), kind, facts_numbers, rng)
                    if new is None or new == m.group(0):
                        continue
                    bad = answer[:m.start()] + new + answer[m.end():]
                    # the guard normalises tokens ("120.90" -> "120.9"), so compare values
                    caught = any(float(t) == abs(float(new)) for t in unsupported_numbers(bad, facts))
                    trials[kind]["n"] += 1
                    trials[kind]["flagged"] += caught
                    if kind in misses and not caught and len(misses[kind]) < 5:
                        misses[kind].append({"portfolio": name, "original": m.group(0),
                                             "written": new, "line": answer})

    rate = {k: (v["flagged"] / v["n"] if v["n"] else None) for k, v in trials.items()}
    caught_n = sum(trials[k]["flagged"] for k in CATCH)
    total_n = sum(trials[k]["n"] for k in CATCH)
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "tolerance": 0.05, "seed": SEED, "fact_blocks": list(blocks),
           "false_positive_rate": {"verbatim": rate["verbatim"], "rounded": rate["rounded"]},
           "detection_rate": {k: rate[k] for k in CATCH},
           "detection_rate_all_catchable": caught_n / total_n,
           "blind_spot_detection_rate": {k: rate[k] for k in BLIND},
           "trials": trials, "example_misses": misses}
    OUT_PATH.write_text(json.dumps(out, indent=2))

    print(f"Number guard, {len(blocks)} real fact blocks, tolerance 5%")
    print("  correct answers wrongly flagged:")
    for k in ("verbatim", "rounded"):
        print(f"    {k:<14} {trials[k]['flagged']:>3}/{trials[k]['n']:<4} ({rate[k]:.1%})")
    print(f"  invented figures caught: {caught_n}/{total_n} ({caught_n / total_n:.1%})")
    for k in CATCH:
        print(f"    {k:<14} {trials[k]['flagged']:>3}/{trials[k]['n']:<4} ({rate[k]:.1%})")
    print("  blind spots (expected to pass the guard):")
    for k in BLIND:
        print(f"    {k:<14} {trials[k]['flagged']:>3}/{trials[k]['n']:<4} ({rate[k]:.1%} caught)")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
