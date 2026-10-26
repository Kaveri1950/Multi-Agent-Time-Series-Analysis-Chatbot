"""Does Box Least Squares recover NASA's published period? Every cached target.

An earlier report draft stated that BLS recovers the published period within
1% for 60.9% of the 440 downloaded Kepler targets, but no result file was
ever saved for it. This runs astronomy.transit_search on all of them, in
parallel, and writes the evidence.

  significant peak   SDE >= 7, the usual detection threshold
  within 1%          found period within 1% of P, 2P, P/2, 3P or P/3
  exact              within 1% of P itself, not a harmonic

Run:  python -m evaluation.bls_recovery          (all 440; about an hour on 8 cores)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).parent / "results" / "bls_recovery.json"
SDE_THRESHOLD = 7.0


def one(kepid: int, label: str) -> dict:
    from astronomy.bulk_download import load_target
    from astronomy.koi_catalog import fetch_koi_table
    from astronomy.transit_search import compare_with_truth, search

    try:
        df = load_target(kepid)
        res = search(df["time"].to_numpy(), df["flux"].to_numpy())
        cmp = compare_with_truth(res, kepid, fetch_koi_table())
        return {"kepid": kepid, "label": label, "found_period": res["period"],
                "true_period": cmp.get("true_period"), "harmonic": cmp.get("harmonic"),
                "period_error_pct": cmp.get("period_error_pct"), "sde": res["sde"],
                "within_1pct": bool(cmp.get("matched")), "exact": bool(cmp.get("exact_match"))}
    except Exception as exc:  # noqa: BLE001
        return {"kepid": kepid, "label": label, "error": f"{type(exc).__name__}: {exc}"}


def main():
    from joblib import Parallel, delayed

    from astronomy.bulk_download import load_manifest

    man = load_manifest()
    rows = Parallel(n_jobs=-1, verbose=5)(delayed(one)(int(r.kepid), r.label)
                                          for r in man.itertuples())
    d = pd.DataFrame(rows)
    ok = d[d.get("error").isna()] if "error" in d else d
    sig = ok["sde"] >= SDE_THRESHOLD

    def rates(s):
        return {"n": int(len(s)), "within_1pct": float(s["within_1pct"].mean()),
                "exact": float(s["exact"].mean()),
                "significant_peak": float((s["sde"] >= SDE_THRESHOLD).mean())}

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_targets": int(len(d)), "n_errors": int(len(d) - len(ok)),
           "sde_threshold": SDE_THRESHOLD, "all": rates(ok),
           "by_label": {lab: rates(g) for lab, g in ok.groupby("label")},
           "harmonic_share_of_within_1pct": float(
               (ok["within_1pct"] & ~ok["exact"]).sum() / max(ok["within_1pct"].sum(), 1)),
           "median_sde": float(ok["sde"].median()),
           "median_period_error_pct_when_within_1pct": float(
               ok.loc[ok["within_1pct"], "period_error_pct"].median()),
           "rows": d.replace({np.nan: None}).to_dict("records")}
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))

    a = out["all"]
    print(f"\nBLS on {a['n']} Kepler targets ({out['n_errors']} errors)")
    print(f"  significant peak (SDE >= 7): {a['significant_peak']:.1%}")
    print(f"  period within 1% (incl. harmonics): {a['within_1pct']:.1%}")
    print(f"  exact period: {a['exact']:.1%}")
    for lab, r in out["by_label"].items():
        print(f"    {lab:16} within 1% {r['within_1pct']:.1%}  exact {r['exact']:.1%}  (n={r['n']})")
    print(f"  median SDE {out['median_sde']:.1f}; harmonic share of matches "
          f"{out['harmonic_share_of_within_1pct']:.1%}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
