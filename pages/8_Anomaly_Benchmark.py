"""Cross-domain anomaly detector comparison.

Follows the same shape as the other domain pages -- headline result, then
the evidence, then the honest limits -- with one addition specific to this
experiment: the dataset audit sits BEFORE the results rather than after.

That ordering is the argument. Wu & Keogh's conclusion about the standard
benchmarks is that on flawed data "there is simply no level of performance
that would suggest the utility of a proposed algorithm", so a reader has no
reason to care about the detector table until they know whether the datasets
deserve to be believed. Putting the audit first makes that dependency
visible instead of burying it in a limitations paragraph nobody reaches.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from anomaly.explain import (annotate_flagged, feature_deviations,
                             suppress_overlapping)
from dashboard.charts import (audit_verdict_table, detector_domain_scatter,
                              detector_grid_bar, dual_series_anomaly_chart,
                              feature_deviation_bar,
                              series_anomaly_chart, window_zoom_chart)

RESULTS_PATH = Path(__file__).parent.parent / "evaluation" / "results" / "anomaly_grid.json"

st.title("🔍 Anomaly Benchmark — One Method, Two Domains")
st.caption("Twelve detectors run unchanged on Kepler light curves and Indian equity returns. "
           "Same code, refitted separately on each dataset — method generalisation, not transfer "
           "learning. Scored with PR-AUC against no-skill floors, with no point adjustment.")


@st.cache_data(show_spinner=False)
def _load():
    if not RESULTS_PATH.exists():
        return None
    return json.loads(RESULTS_PATH.read_text())


results = _load()
if results is None:
    st.error("No benchmark results found. Run this once, then reload:\n\n"
             "```\npython -m anomaly.grid --kepler-limit 60\n```\n\n"
             "A quick check that reproduces the same ordering in ~2 minutes: "
             "`python -m anomaly.grid --kepler-limit 12`")
    st.stop()

rows = results["rows"]
audits = results.get("audits", {})
summaries = results.get("summaries", {})
meta = results.get("meta", {})
domains = sorted({r["domain"] for r in rows})


# ==========================================================================
# 1. the headline -- did anything beat a detector that learned nothing?
# ==========================================================================

st.subheader("Did anything actually beat doing nothing?")

st.markdown(
    "Every detector is measured against **no-skill floors** — a rule that uses the raw number "
    "with no fitting at all, and an autoencoder with random untrained weights. "
    "Kim et al. (AAAI 2022) found untrained models matching *published* detectors on standard "
    "benchmarks, so clearing this bar is the minimum evidence that a method learned anything."
)

cols = st.columns(len(domains))
for col, domain in zip(cols, domains):
    s = summaries.get(domain, {})
    best, floor = s.get("best_real_value"), s.get("best_floor_value")
    margin = s.get("margin_over_floor")
    beat = s.get("real_beats_floor")
    with col:
        st.metric(
            f"{domain} — best detector",
            f"{best:.4f}" if best is not None else "n/a",
            delta=(f"{margin:+.4f} vs floor" if margin is not None else None),
            delta_color="normal" if beat else "off",
            help=f"{s.get('best_real')} vs best floor {s.get('best_floor')} ({floor})",
        )
        frac = s.get("fraction_of_ceiling")
        if frac is not None:
            st.caption(
                f"**{frac:.0%}** of the supervised ceiling"
                + ("  ·  a model *with* the labels does worse" if frac > 1 else "")
            )

# The two domains disagreed, and that disagreement is the finding -- so it is
# stated as prose rather than left for the reader to infer from two numbers.
eq, kep = summaries.get("equity", {}), summaries.get("kepler", {})
if eq and kep:
    st.info(
        f"**The two domains gave opposite answers.** On Kepler the best detector "
        f"(`{kep.get('best_real')}`) clears the floor by **{kep.get('margin_over_floor'):+.4f}** "
        f"and recovers **{kep.get('fraction_of_ceiling', 0):.0%}** of what a label-aware model "
        f"achieves — a learned representation genuinely helps. On equities the margin is "
        f"**{eq.get('margin_over_floor'):+.4f}**, which is nothing, and the supervised ceiling "
        f"scores *below* both. So whether sophistication pays is a property of the domain, "
        f"not of the method — and the only way to find out is to measure against a floor.",
        icon="🔍",
    )


# ==========================================================================
# 2. inspect -- what did it actually flag, and why?
# ==========================================================================

st.divider()
st.subheader("Inspect the detections")

st.markdown(
    "Everything above is a scoreboard. This section runs the winning detector on a single "
    "series and shows what it actually flagged, why, and — crucially — **where it was wrong**. "
    "Reasons are rule-based, derived from the same features the detector saw. No language "
    "model is involved in producing them, so the explanation is one the detector can be held to."
)

BEST = {d: summaries.get(d, {}).get("best_real") for d in domains}

# Kepler first: the light curve is the flagship visual on this page, and
# `sorted()` would otherwise default to equity purely because "e" < "k".
pick = st.selectbox(
    "Series", domains,
    index=domains.index("kepler") if "kepler" in domains else 0,
    format_func=lambda d: {
        "kepler": "Kepler light curve — pick a target",
        "equity": "Indian equity index — pick a date range"}.get(d, d),
)

# ---- per-domain selection ------------------------------------------------
kepid, date_start, date_end = None, None, None

if pick == "kepler":
    from astronomy.bulk_download import load_manifest

    man = load_manifest()
    fcol, tcol = st.columns([1, 3])
    with fcol:
        disp_filter = st.selectbox(
            "Disposition", ["All", "CONFIRMED", "FALSE POSITIVE"],
            help="NASA's verdict on the candidate. Worth looking at both: a detector that "
                 "only ever fires on CONFIRMED targets is not being tested very hard.",
        )
    shown = man if disp_filter == "All" else man[man["label"] == disp_filter]
    with tcol:
        choice = st.selectbox(
            f"Target ({len(shown)} available)", shown.index,
            format_func=lambda i: f"KIC {int(man.loc[i, 'kepid'])} — {man.loc[i, 'label']} "
                                  f"· {int(man.loc[i, 'n_points']):,} points "
                                  f"· {man.loc[i, 'span_days']:.0f} days",
        )
    kepid = int(man.loc[choice, "kepid"])

else:
    # Full panel range, read from the benchmark metadata rather than hardcoded
    # so the widget cannot drift out of step with the cached data.
    em = meta.get("equity", {})
    lo = pd.Timestamp(em.get("first", "2019-01-02")).date()
    hi = pd.Timestamp(em.get("last", "2026-09-21")).date()
    dcol1, dcol2, dcol3 = st.columns([2, 2, 3])
    with dcol1:
        date_start = st.date_input("From", value=lo, min_value=lo, max_value=hi)
    with dcol2:
        date_end = st.date_input("To", value=hi, min_value=lo, max_value=hi)
    with dcol3:
        preset = st.selectbox(
            "Or jump to", ["— custom —", "COVID crash (2020)", "Adani–Hindenburg (2023)",
                           "Election result (2024)", "Full history"],
            help="Shortcuts to periods with known events in the label calendar.",
        )
        presets = {
            "COVID crash (2020)": ("2020-01-01", "2020-06-30"),
            "Adani–Hindenburg (2023)": ("2022-11-01", "2023-04-30"),
            "Election result (2024)": ("2024-03-01", "2024-09-30"),
            "Full history": (str(lo), str(hi)),
        }
        if preset in presets:
            date_start = pd.Timestamp(presets[preset][0]).date()
            date_end = pd.Timestamp(presets[preset][1]).date()
            st.caption(f"→ {date_start} to {date_end}")

    if date_start >= date_end:
        st.error("The start date must come before the end date.")
        st.stop()

detector_name = st.selectbox(
    "Detector", [r["detector"] for r in rows if r["domain"] == pick and not r.get("error")],
    index=0 if not BEST.get(pick) else
    max(0, [r["detector"] for r in rows if r["domain"] == pick].index(BEST[pick])),
    help="Defaults to the best-performing detector on this domain.",
)

contamination = st.slider(
    "Flag the top N% of windows", 0.5, 20.0, 2.0, 0.5,
    help="These detectors produce a SCORE, not a yes/no. Where the cut falls is a choice, "
         "and moving it trades false positives against misses — which is exactly why the "
         "benchmark above is scored threshold-free.",
) / 100.0


@st.cache_data(show_spinner="Running detector...", max_entries=12)
def _inspect(domain: str, det_name: str, kepid: int | None,
             date_start=None, date_end=None):
    """Score one series live.

    The date range is applied to the SERIES BEFORE windowing and fitting, not
    as a filter on results afterwards. That is the honest reading of "look at
    2020": the detector sees only that period, and works out what normal
    looks like from it. Filtering afterwards would let the rest of history
    define normal, which is a different -- and, for a causal detector,
    dishonest -- question.

    Cached because nudging the threshold slider must not refit anything; the
    cut moves, the scores do not.
    """
    from anomaly.detectors import default_detectors
    from anomaly.grid import WINDOW_CONFIG

    cfg = WINDOW_CONFIG[domain]
    det = next(d for d in default_detectors(include_finance_only=True) if d.name == det_name)

    if domain == "equity":
        from anomaly.core import windowize
        from anomaly.labels import finance_day_labels
        from quant.data import load_panel

        close, volume = load_panel()
        tv = (close * volume).rolling(21, min_periods=10).mean()
        liquid = tv.mean().nlargest(50).index
        series = close[liquid].pct_change().mean(axis=1).dropna()
        if date_start is not None:
            series = series.loc[str(date_start):str(date_end)]
        if len(series) < cfg["window"] * 3:
            raise ValueError(
                f"only {len(series)} trading days in this range -- need at least "
                f"{cfg['window'] * 3} for a {cfg['window']}-day window to mean anything"
            )
        x, y_vals = series.index, series.to_numpy()
        windows = windowize(y_vals, cfg["window"], cfg["stride"])
        day_labels = finance_day_labels(series.index)
        y_true = day_labels[cfg["window"] - 1:][: len(windows)]
        windows = windows[: len(y_true)]
        label = "daily return (equal-weighted, 50 most liquid)"
        # The index LEVEL, rebased to 100 at the start of the selected range.
        # This is the series a person recognises; the detector never sees it.
        real = (1.0 + series).cumprod() * 100.0
        real_label = "equal-weighted index level (rebased to 100)"
    else:
        from anomaly.core import windowize
        from anomaly.labels import astronomy_window_labels
        from astronomy.bulk_download import load_target
        from astronomy.koi_catalog import fetch_koi_table

        df = load_target(int(kepid))
        t, f = df["time"].to_numpy(), df["flux"].to_numpy()
        ok = np.isfinite(t) & np.isfinite(f)
        x, y_vals = t[ok], f[ok]
        windows = windowize(y_vals, cfg["window"], cfg["stride"])
        y_true = astronomy_window_labels(x, int(kepid), cfg["window"], cfg["stride"],
                                         fetch_koi_table())
        n = min(len(windows), len(y_true))
        windows, y_true = windows[:n], y_true[:n]
        label = "normalised flux (what the detector saw)"
        # Parts per million from the star's own baseline -- the unit
        # astronomers actually read transit depths in. Raw electrons/sec is
        # not recoverable: bulk_download applies .normalize() before caching
        # and does not keep the divisor.
        real = (y_vals - 1.0) * 1e6
        real_label = "brightness deviation (parts per million)"

    scores = det.fit_score(windows, y_true)
    return x, y_vals, windows, y_true, scores, cfg, label, real, real_label


try:
    (x, y_vals, windows, y_true, scores, cfg, y_label,
     real_y, real_label) = _inspect(
        pick, detector_name, kepid, date_start, date_end)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not run the detector on this series: {exc}")
    st.stop()

if int(np.sum(y_true)) == 0:
    st.warning(
        "**No labelled events fall in this selection**, so precision and the verdict column "
        "cannot be computed — every flag will read as a false positive by default. The "
        "detector still runs and the reasons are still valid; there is simply nothing to "
        "check them against. Widen the range to include a known event.",
        icon="⚠️",
    )

cut = np.quantile(scores, 1 - contamination)
flagged = scores >= cut

# Overlapping windows mean a single event produces a RUN of identical top
# scores -- at stride 1, one extreme day is the maximum of 21 consecutive
# windows. Counting raw flags would report 21 detections where there is one
# event, so distinct events are counted with the same non-maximum
# suppression the detections table uses. The two must agree.
n_flagged = int(flagged.sum())
distinct = len(suppress_overlapping(np.where(flagged, scores, -np.inf),
                                    top_n=n_flagged, window=cfg["window"],
                                    stride=cfg["stride"])) if n_flagged else 0
hits = int(np.sum(flagged & (y_true == 1)))
prec = hits / max(n_flagged, 1)
chance = float(np.mean(y_true))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Windows", f"{len(windows):,}")
m2.metric("Flagged", f"{n_flagged:,}",
          help=f"Top {contamination:.1%} by score. These overlap: they cover about "
               f"{distinct} distinct events.")
m3.metric("Distinct events flagged", f"{distinct:,}",
          help="Overlapping flags merged, so this matches the detections table below "
               "and the shaded bands on the chart.")
m4.metric("Precision", f"{prec:.1%}",
          delta=f"{prec - chance:+.1%} vs chance" if chance else None,
          delta_color="normal",
          help="Of the flagged windows, how many were genuinely anomalous. Chance is the "
               "base rate. Computed window-wise, the standard convention.")

show_real = st.toggle(
    "Show the series in real units too", value=True,
    help="The transformation that makes a series analysable also makes it unreadable. "
         "The top panel is what a person recognises; the bottom is what the detector "
         "actually operated on. Same shading on both.",
)
if show_real:
    st.plotly_chart(
        dual_series_anomaly_chart(x, real_y, y_vals, cfg["window"], cfg["stride"],
                                  flagged, y_true, real_label=real_label,
                                  model_label=y_label,
                                  title=f"{detector_name} on {pick}"),
        use_container_width=True,
    )
else:
    st.plotly_chart(
        series_anomaly_chart(x, y_vals, cfg["window"], cfg["stride"], flagged, y_true,
                             title=f"{detector_name} on {pick}", y_label=y_label),
        use_container_width=True,
    )

st.markdown("##### Top detections, with reasons")
st.caption("Sorted by score. `verdict` compares each flag against the independent label set — "
           "false positives are shown rather than filtered out.")

annotated = annotate_flagged(windows, scores, pick, y_true, top_n=15,
                             window=cfg["window"], stride=cfg["stride"])
display = annotated.assign(reasons=annotated["reasons"].map(lambda r: " · ".join(r)))
st.dataframe(
    display[["rank", "window", "score", "verdict", "likely_cause", "reasons"]],
    use_container_width=True, hide_index=True,
)

st.markdown("##### Look at one")
row_idx = st.selectbox(
    "Detection", annotated.index,
    format_func=lambda i: f"#{annotated.loc[i, 'rank']} — {annotated.loc[i, 'likely_cause']} "
                          f"({annotated.loc[i, 'verdict']})",
)
chosen = annotated.loc[row_idx]

zl, zr = st.columns([3, 2])
with zl:
    st.plotly_chart(
        window_zoom_chart(x, y_vals, int(chosen["start_index"]), cfg["window"],
                          title=f"Window {int(chosen['window'])} in context", y_label=y_label),
        use_container_width=True,
    )
with zr:
    st.plotly_chart(
        feature_deviation_bar(feature_deviations(windows, int(chosen["window"])).iloc[0]),
        use_container_width=True,
    )

verdict_icon = "✅" if chosen.get("verdict") == "true positive" else "❌"
st.markdown(f"**{verdict_icon} {chosen['likely_cause']}** — {chosen['explanation']}")
st.markdown("**Why it was flagged:**")
for reason in chosen["reasons"]:
    st.markdown(f"- {reason}")

if chosen.get("verdict") == "false positive":
    st.caption("This window scored highly but no label supports it. On equity that may mean the "
               "event calendar is incomplete rather than that the detector was wrong; on Kepler "
               "it means NASA's ephemerides predict no transit here.")


# ==========================================================================
# 3. the audit -- before the results, on purpose
# ==========================================================================

st.divider()
st.subheader("Do these datasets deserve to be believed?")

st.markdown(
    "Wu & Keogh (IEEE TKDE) audited the standard Yahoo / Numenta / NASA benchmarks and found "
    "most of their problems suffer from at least one of four flaws — concluding that on such "
    "data *“there is simply no level of performance that would suggest the utility of a "
    "proposed algorithm.”* Almost nobody runs that audit on their own data. These two datasets "
    "were audited before any detector result below was reported."
)

st.plotly_chart(audit_verdict_table(audits), use_container_width=True)

fcols = st.columns(len(audits))
for col, (domain, audit) in zip(fcols, audits.items()):
    with col:
        st.markdown(f"**{domain}**")
        for check in audit["checks"]:
            verdict = check.get("verdict", "—")
            icon = {"pass": "✅", "warn": "⚠️", "fail": "❌",
                    "review": "⚠️"}.get(verdict, "➖")
            detail = ", ".join(f"`{k}`={v}" for k, v in check.items()
                               if k not in ("flaw", "verdict"))
            with st.expander(f"{icon} {check['flaw'].replace('_', ' ')}", expanded=False):
                st.caption(detail)

with st.expander("Why the audit comes first"):
    st.markdown(
        "- **Triviality** — if one line of arithmetic solves it, a deep model succeeding "
        "proves nothing. The one-liner detectors below *are* this test.\n"
        "- **Anomaly density** — real anomalies are rare. When half the data is labelled "
        "anomalous the task has quietly become classification.\n"
        "- **Run-to-failure bias** — in many benchmarks anomalies cluster at the end, so "
        "“flag the last point” scores well for no reason.\n"
        "- **Label quality** — labels that disagree with the data cap every detector's "
        "achievable score, so a low number may be the labels' fault rather than the method's."
    )


# ==========================================================================
# 4. the results
# ==========================================================================

st.divider()
st.subheader("The detector grid")

metric = st.radio(
    "Metric", ["pr_auc", "roc_auc", "lift"], horizontal=True,
    format_func=lambda m: {"pr_auc": "PR-AUC (recommended)", "roc_auc": "ROC-AUC",
                           "lift": "Lift over chance"}[m],
    help="PR-AUC is floored at the base rate, so it stays interpretable when anomalies are "
         "rare. ROC-AUC's floor is always 0.5 no matter how imbalanced the data, which is "
         "why it can make a weak detector look respectable.",
)

tabs = st.tabs([d.capitalize() for d in domains])
for tab, domain in zip(tabs, domains):
    with tab:
        m = meta.get(domain, {})
        st.caption(" · ".join(f"**{k}**: {v}" for k, v in m.items()))
        st.plotly_chart(detector_grid_bar(rows, domain, metric), use_container_width=True)

        table = pd.DataFrame([r for r in rows if r["domain"] == domain])
        keep = [c for c in ["detector", "group", "roc_auc", "pr_auc", "lift",
                            "prevalence", "seconds"] if c in table.columns]
        st.dataframe(
            table[keep].sort_values("pr_auc", ascending=False),
            use_container_width=True, hide_index=True,
        )


# ==========================================================================
# 5. the generalisation question
# ==========================================================================

st.divider()
st.subheader("Does a detector that works on one domain work on the other?")

st.markdown(
    "The claim being tested is deliberately modest: **not** that a model trained on stars "
    "transfers to stocks — that is transfer learning, and a much harder thing to defend — but "
    "that the same *method*, refitted independently on each dataset, works on both. "
    "Shown as lift over chance so two domains with very different base rates can share a plot."
)

st.plotly_chart(detector_domain_scatter(rows), use_container_width=True)


# ==========================================================================
# 6. the honest limits
# ==========================================================================

st.divider()
st.subheader("What these numbers do not say")

kepler_audit = audits.get("kepler", {})
quiet = next((c.get("fraction_quiet_positives") for c in kepler_audit.get("checks", [])
              if c["flaw"] == "mislabeled_ground_truth"), None)

st.warning(
    "**The ceiling is low because of the representation, not the detectors.** "
    + (f"On Kepler, **{quiet:.0%}** of labelled transits are no more extreme than a typical "
       "non-transit window at this window size. " if quiet else "")
    + "Even a model allowed to see the labels reaches only modest lift, which means most "
    "labelled transits are effectively invisible in these summary statistics. A different "
    "window length or a phase-aware representation would move that ceiling; a better "
    "detector on these features cannot.",
    icon="⚠️",
)

st.markdown(
    "- **The equity label set is incomplete by construction.** It is a short, pre-registered "
    "list of events that happened for reasons of their own — budget sessions, election "
    "results, documented shocks. Real shocks are missing from it, so a detector penalised "
    "for flagging a genuine event absent from the list is being penalised for being right. "
    "Recall therefore reads low; precision-style metrics read honestly.\n"
    "- **Kepler's anomaly density warns.** Short-period planets transit often, so NASA's "
    "ephemerides genuinely produce a high positive rate — but it sits above the range Wu & "
    "Keogh consider realistic, and the comparison should be read with that in mind.\n"
    "- **One configuration, no tuning.** Every detector ran once at a pre-declared setting. "
    "Metrics gathered by trying many configurations and reporting the best become "
    "uninterpretable, so the reporting budget is fixed at one and stated.\n"
    "- **No point adjustment.** The dominant protocol in this literature marks a whole "
    "anomaly segment correct if any moment inside it is flagged, under which a *random* score "
    "reaches state-of-the-art F1. Nothing here uses it."
)

with st.expander("Reproduce this"):
    st.code("python -m anomaly.grid --kepler-limit 60   # full run, ~52 min\n"
            "python -m anomaly.grid --kepler-limit 12   # same ordering, ~2 min", language="bash")
    st.caption(f"Results read from `{RESULTS_PATH.relative_to(RESULTS_PATH.parent.parent.parent)}`")
