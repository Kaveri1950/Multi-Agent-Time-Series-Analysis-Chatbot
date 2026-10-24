"""How well does filing search find the right kind of filing?

LABELS COME FROM THE EXCHANGE, NOT FROM ME. Every NSE announcement carries a
category chosen when it was filed ("Credit Rating", "Acquisition", "Cessation"
...). Only the filing text is indexed; the category is never embedded. Each
query below is a plain-English description of an event, written without
copying the category name where possible, and a retrieved filing counts as
relevant if its exchange category is in the query's set.

That makes the scores a LOWER BOUND: a credit-rating letter filed under the
generic "Updates" category is relevant but counts as a miss. It also means
the queries are mine, so the test measures how well the retrievers map
everyday wording onto the exchange's taxonomy -- the job a search box does.

Metrics, per query, then averaged:
  precision@10   share of the top 10 in the right category
  MRR            1 / rank of the first relevant filing
  base rate      share of the whole index in the right category (no-skill floor)

Run:  python -m evaluation.filing_search_eval     (needs the index: python -m finance.filing_search --build)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT_PATH = Path(__file__).parent / "results" / "filing_search.json"
K = 10
METHODS = ("keyword", "dense", "hybrid")

QUERIES = [
    ("credit rating agency revised or reaffirmed the company's debt rating",
     {"Credit Rating", "Credit Rating- Revision", "Credit Rating- New", "Credit Rating- Others"}),
    ("company is buying another business or taking a stake in it",
     {"Acquisition", "Update-Acquisition/Scheme/Sale/Disposal/Reg30-XBRL"}),
    ("an independent director stepped down from the board",
     {"Resignation", "Resignation of Independent director", "Cessation",
      "Resignation of Director/KMP/SMP", "Change in Director(s)", "Retirement"}),
    ("shareholder lost physical share certificates, duplicate requested",
     {"Loss of Share Certificates", "Loss of share certificate", "Issue of Duplicate Share Certificate"}),
    ("company will repurchase its own shares from shareholders",
     {"Buyback", "Public Announcement - Buyback of Shares", "Post Buyback Public Announcement",
      "Closure of Buy Back", "Buyback - others", "Buyback - Open Market"}),
    ("interim dividend declared per equity share",
     {"Dividend", "Dividend Updates", "Date of payment of dividend"}),
    ("merger of group companies under a scheme sanctioned by the tribunal",
     {"Amalgamation/Merger", "Scheme of Arrangement", "NCLT/ Court Convened Meeting", "Demerger",
      "Other Restructuring", "Restructuring", "Corp Restructuring - others"}),
    ("won a large new order or contract from a customer",
     {"Bagging/Receiving of orders/contracts", "Bagging orders/contract", "Awarding orders/contract",
      "Awarding of order(s)/contract(s)"}),
    ("stock options granted to employees under the incentive plan",
     {"ESOP/ESOS/ESPS", "Allotment of ESOP/ESPS", "ESOP/ESPS/SBEB Scheme"}),
    ("exchange sought clarification on a media report about the company",
     {"News Verification", "News Clarification", "Clarification", "Rumour Verification - Regulation 30(11)",
      "Reply to Clarification Sought", "Price movement", "Spurt in Volume"}),
    ("statutory auditors of the company changed",
     {"Change in Auditors", "Resignation of Statutory Auditor"}),
    ("window for insiders to trade in the shares is closed until results",
     {"Trading Window", "Closure of trading window"}),
    ("regulator imposed a penalty or passed an order against the company",
     {"Action(s) taken or orders passed", "Action(s) initiated or orders passed",
      "Litigations/Disputes/Regulatory actions",
      "Pendency of Litigation(s)/dispute(s) or the outcome impacting the Company"}),
    ("transcript of the earnings conference call with analysts",
     {"Transcript of Analysts/Institutional Investor Meet/Con. Call",
      "Analysts/Institutional Investor Meet/Con. Call Updates",
      "Recording of Analysts/Institutional Investor Meet/Con. Call",
      "Schedule of Analysts/Institutional Investor Meet/Con. Call"}),
    ("issue of bonus shares or split of the face value",
     {"Bonus", "Stock split", "Split of shares"}),
    ("new chief financial officer or senior management appointed",
     {"Change in Management", "Appointment",
      "Change in Directors/ Key Managerial Personnel/ Auditor/ Compliance Officer/ Share Transfer Agent",
      "Change in Company Secretary/Compliance Officer"}),
    ("raising money from institutions through a share placement",
     {"Qualified Institutional Placement", "Raising of Funds", "Issue of Securities",
      "Preferential issue", "Preferential Issue", "Rights Issue"}),
    ("strike or fire disrupted operations at a plant",
     {"Strikes/Lockouts/Disturbances", "Disruption of Operations", "Disruption of operations",
      "Closure of operations"}),
]


def main():
    from finance.filing_search import _index, _rank

    d = _index()[0]
    types = d["type"].to_numpy()
    per_query, summary = [], {m: {"p10": [], "mrr": []} for m in METHODS}
    for q, rel in QUERIES:
        row = {"query": q, "categories": sorted(rel),
               "relevant_in_index": int(np.isin(types, list(rel)).sum()),
               "base_rate": float(np.isin(types, list(rel)).mean())}
        for m in METHODS:
            ids = _rank(q, m, 200)
            hits = [types[i] in rel for i in ids]
            p10 = float(np.mean(hits[:K]))
            first = next((r for r, h in enumerate(hits) if h), None)
            mrr = 0.0 if first is None else 1.0 / (first + 1)
            row[m] = {"p_at_10": p10, "mrr": mrr, "top3": [d.at[i, "text"][:140] for i in ids[:3]]}
            summary[m]["p10"].append(p10)
            summary[m]["mrr"].append(mrr)
        per_query.append(row)

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "index_size": int(len(d)), "n_queries": len(QUERIES), "k": K,
           "label_source": "NSE announcement category (not indexed)",
           "mean_base_rate": float(np.mean([r["base_rate"] for r in per_query])),
           "methods": {m: {"mean_p_at_10": float(np.mean(v["p10"])),
                           "mean_mrr": float(np.mean(v["mrr"])),
                           "queries_with_p10_at_least_half": int(sum(p >= 0.5 for p in v["p10"]))}
                       for m, v in summary.items()},
           "queries": per_query}
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))

    print(f"Filing search: {len(QUERIES)} queries over {len(d):,} unique filings, "
          f"labels = NSE category (never indexed)")
    print(f"  no-skill base rate (mean share of index in the right category): "
          f"{out['mean_base_rate']:.2%}")
    for m, v in out["methods"].items():
        print(f"  {m:<8} precision@10 {v['mean_p_at_10']:.1%}   MRR {v['mean_mrr']:.3f}   "
              f"queries >= 50% precise {v['queries_with_p10_at_least_half']}/{len(QUERIES)}")
    print("\n  per query precision@10 (keyword / dense / hybrid):")
    for r in per_query:
        print(f"   {r['keyword']['p_at_10']:.1f} / {r['dense']['p_at_10']:.1f} / "
              f"{r['hybrid']['p_at_10']:.1f}   {r['query']}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
