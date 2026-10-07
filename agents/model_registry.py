"""A machine-readable catalogue of every model in the project.

This is the piece that turns the system from a router with hardcoded
pipelines into something that actually chooses. Today a finance question --
even "what is Reliance's RSI?", a number already sitting in a dataframe --
runs ARIMA, XGBoost, an RSS fetch, FinBERT over every headline, a SECOND
XGBoost fit with sentiment, and five anomaly detectors, because
agents/graph.py wires finance -> sentiment -> finance_anomaly -> explain as
unconditional edges. Nothing selects; everything runs.

THE DESIGN IDEA: select on TASK, not on domain.

Domain says which data. Task says which model. They are independent axes,
and separating them is what stops this becoming a large if/else:

    query -> (domain, task) -> registry -> the model

Adding a model then means adding one entry here, not editing routing code --
which is the generality claim the thesis makes, expressed as something a
reader can check.

WHY THE SKILL FIELD IS POPULATED HONESTLY: the selector shows these numbers
to the LLM when it chooses, and they also appear in answers. A registry that
advertised the autoencoder without recording that it scores at chance would
cause the system to pick it and then describe its output as meaningful. The
measured limitations are load-bearing, not decoration.
"""

from typing import Callable, TypedDict


# --------------------------------------------------------------------------
# task taxonomy -- the axis that cuts across domains
# --------------------------------------------------------------------------

FORECAST_VALUE = "forecast_value"                    # predict a number
FORECAST_EVENT_PROBABILITY = "forecast_event_prob"   # P(event in a window)
FORECAST_EVENT_COUNT = "forecast_event_count"        # how many events
DETECT_ANOMALY = "detect_anomaly"                    # which points are odd
EVALUATE_MODEL = "evaluate_model"                    # how good is the model
COMPARE_MODELS = "compare_models"                    # does X help
SCORE_SENTIMENT = "score_sentiment"                  # text -> score
LOOKUP_FACT = "lookup_fact"                          # authoritative record
EXTRACT_SERIES = "extract_series"                    # image -> time series
ASSESS_RISK = "assess_risk"                          # how much could this lose
FORECAST_VOLATILITY = "forecast_volatility"          # how big the moves will be

TASKS = [
    FORECAST_VALUE, FORECAST_EVENT_PROBABILITY, FORECAST_EVENT_COUNT,
    DETECT_ANOMALY, EVALUATE_MODEL, COMPARE_MODELS, SCORE_SENTIMENT,
    LOOKUP_FACT, EXTRACT_SERIES, ASSESS_RISK, FORECAST_VOLATILITY,
]

TASK_DESCRIPTIONS = {
    FORECAST_VALUE: "predict a future numeric value (a price, a NAV)",
    FORECAST_EVENT_PROBABILITY: "the chance an event occurs within a time window",
    FORECAST_EVENT_COUNT: "how many events to expect within a time window",
    DETECT_ANOMALY: "find which points in an existing series are unusual",
    EVALUATE_MODEL: "report how accurate a model is, from backtesting",
    COMPARE_MODELS: "compare two approaches to say whether one helps",
    SCORE_SENTIMENT: "score text as positive/negative",
    LOOKUP_FACT: "retrieve an authoritative recorded fact, no modelling",
    EXTRACT_SERIES: "recover a time series from a plot image",
    ASSESS_RISK: "how much could be lost holding this, and where that risk comes from",
    FORECAST_VOLATILITY: "how big the moves will be, not which direction",
}


class ModelSpec(TypedDict, total=False):
    id: str
    name: str
    domain: str                  # finance | astronomy | solar | any
    task: str
    method_family: str           # statistical | machine_learning | deep_learning | rule | catalogue
    description: str             # shown to the LLM when it chooses
    requires: list               # what must be known before this can run
    returns: str
    skill: dict                  # measured, with the metric named
    when_to_use: str
    when_not_to_use: str         # the field that prevents confident misuse
    cost: str                    # fast | medium | slow
    entry_point: str             # importable path, for the report's appendix


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------

REGISTRY: list[ModelSpec] = [

    # ---------------- finance ----------------
    {
        "id": "finance.arima_forecast",
        "name": "ARIMA forward projection",
        "domain": "finance", "task": FORECAST_VALUE,
        "method_family": "statistical",
        "description": "Projects a price series forward N days with a confidence interval. "
                       "Genuinely forward-looking, not a backtest.",
        "requires": ["ticker", "horizon_days"],
        "returns": "predicted prices + confidence interval + chart",
        "skill": {"metric": "interval coverage", "value": 0.8305,
                  "note": "the close fell inside the 80% interval 83.1% of the time over 118 "
                          "walk-forward days (95% CI 0.750-0.893, p=0.49 against the 80% "
                          "target) -- as well calibrated as the GP, which had this registered "
                          "and ARIMA did not. On POINT accuracy it ties the naive baseline "
                          "(MAE 20.94 vs 20.88, Wilcoxon p=0.79). Failed to fit on 2 of 120 "
                          "days. Per-ticker RMSE is still reported at query time."},
        "when_to_use": "the user asks what a price WILL BE at some future date",
        "when_not_to_use": "the user asks how accurate the model is, or about past behaviour. "
                           "Beyond a few days the horizon is UNEVIDENCED -- every measurement "
                           "behind the skill value above is one step ahead",
        "cost": "medium",
        "entry_point": "finance.price_model.forecast_future",
    },
    {
        "id": "finance.arima_backtest",
        "name": "ARIMA baseline backtest",
        "domain": "finance", "task": EVALUATE_MODEL,
        "method_family": "statistical",
        "description": "Holds out the last 30 days and measures RMSE. Answers how "
                       "trustworthy the forecast is, not what the price will be.",
        "requires": ["ticker"],
        "returns": "RMSE on held-out data",
        "skill": {"metric": "RMSE", "value": "per-ticker"},
        "when_to_use": "the user asks how accurate or reliable the model is",
        "when_not_to_use": "the user wants an actual future prediction",
        "cost": "medium",
        "entry_point": "finance.price_model.arima_baseline",
    },
    {
        "id": "finance.xgboost_sentiment",
        "name": "XGBoost with/without sentiment",
        "domain": "finance", "task": COMPARE_MODELS,
        "method_family": "machine_learning",
        "description": "Trains the same gradient-boosted model twice, with and without "
                       "FinBERT news sentiment, and reports both RMSEs.",
        "requires": ["ticker"],
        "returns": "RMSE with sentiment vs without",
        "skill": {"metric": "RMSE delta", "value": "per-ticker",
                  "note": "single 30-day split, not walk-forward -- a known weakness"},
        "when_to_use": "the user asks whether news/sentiment improves prediction",
        "when_not_to_use": "for a plain forecast -- this is a comparison, and it is slow",
        "cost": "slow",
        "entry_point": "finance.price_model.compare_with_without_sentiment",
    },
    {
        "id": "finance.gpr_logged_forecast",
        "name": "Gaussian Process forecast (pre-computed daily)",
        "domain": "finance", "task": FORECAST_VALUE,
        "method_family": "statistical",
        "description": "Next-day close with a calibrated 80% predictive interval, read from "
                       "the daily forecast log rather than fitted on demand. Always reports the "
                       "naive random-walk baseline alongside it.",
        "requires": ["ticker"],
        "returns": "point forecast + 80% interval + naive baseline + measured track record",
        "skill": {"metric": "interval coverage", "value": 0.825,
                  "note": "WELL CALIBRATED -- the actual close fell inside the 80% interval 82.5% "
                          "of the time over 120 walk-forward days (95% CI 0.745-0.888, p=0.57 against "
                          "the 80% target). But it does NOT beat the naive baseline on point "
                          "accuracy (RMSE ratio 1.22, Wilcoxon p=0.0001) and its directional "
                          "accuracy of 0.44 is NOT distinguishable from chance (95% CI "
                          "0.351-0.535, p=0.24). Use it for the interval, not the point."},
        "when_to_use": "the user wants a next-day price with an honest uncertainty range, or asks "
                       "how confident the forecast is",
        "when_not_to_use": "for a multi-day horizon (the log stores one step ahead), and never as "
                           "grounds for a trading decision -- it cannot beat assuming no change",
        "cost": "fast",
        "entry_point": "finance.forecast_log.latest",
    },
    {
        "id": "finance.har_volatility",
        "name": "HAR realized-volatility forecast",
        "domain": "finance", "task": FORECAST_VOLATILITY,
        "method_family": "statistical",
        "description": "Next-day realized volatility from the daily, weekly and monthly "
                       "averages of past volatility (Corsi's HAR), with volatility estimated "
                       "from the daily high-low range so no paid intraday feed is needed. "
                       "Reports the naive carry-forward baseline alongside it.",
        "requires": ["ticker"],
        "returns": "next-day annualised volatility + naive baseline + measured track record",
        "skill": {"metric": "QLIKE", "value": 0.4105,
                  "note": "BEATS ITS BASELINE, which nothing else in this registry does: naive "
                          "carry-forward scores 0.7820, so HAR cuts the loss 47%, and naive is "
                          "the best model on 0 of 48 tickers. Diebold-Mariano -38.69, p < 1e-16 "
                          "over 59,709 strictly-causal walk-forward forecasts. Reproduces on all "
                          "three range estimators (Garman-Klass 0.340, Rogers-Satchell 0.368). "
                          "Log-HAR is statistically INDISTINGUISHABLE (p=0.12), so it is not "
                          "registered separately."},
        "when_to_use": "the user asks how volatile a stock will be, or wants a volatility or "
                       "risk-of-movement forecast",
        "when_not_to_use": "for a price or a direction -- volatility says how BIG the move will "
                           "be, never which way. Returns themselves stay unforecastable here",
        "cost": "fast",
        "entry_point": "quant.volatility.walk_forward_ticker",
    },
    {
        "id": "finance.anomaly_detectors",
        "name": "Five statistical anomaly detectors",
        "domain": "finance", "task": DETECT_ANOMALY,
        "method_family": "statistical",
        "description": "Rolling z-scores for return outliers, volatility spikes, volume "
                       "spikes, overnight gaps, and sentiment-price divergence. Each flag "
                       "carries a computed reason, so days are explained not just flagged.",
        "requires": ["ticker"],
        "returns": "typed, severity-scored anomalous days with reasons",
        "skill": {"metric": "none", "value": None,
                  "note": "NO ground-truth evaluation exists yet -- validated only "
                          "anecdotally on 2024-06-04, India's election result day"},
        "when_to_use": "the user asks what was unusual, odd, or anomalous about a stock",
        "when_not_to_use": "for prediction -- this describes the past only",
        "cost": "medium",
        "entry_point": "finance.anomaly_detector.detect_all",
    },
    {
        "id": "finance.finbert_sentiment",
        "name": "FinBERT news sentiment",
        "domain": "finance", "task": SCORE_SENTIMENT,
        "method_family": "deep_learning",
        "description": "Scores free RSS and Reddit headlines with FinBERT into a daily "
                       "sentiment series.",
        "requires": ["ticker"],
        "returns": "scored headlines + daily sentiment series",
        "skill": {"metric": "none", "value": None,
                  "note": "coverage is thin -- RSS reaches back only weeks, so older "
                          "days fill with 0.0"},
        "when_to_use": "the user asks about news, sentiment, or market mood",
        "when_not_to_use": "for historical sentiment beyond a few weeks ago",
        "cost": "slow",
        "entry_point": "finance.sentiment.get_ticker_sentiment",
    },
    {
        "id": "finance.mutual_fund_nav",
        "name": "Mutual fund NAV forecast",
        "domain": "finance", "task": FORECAST_VALUE,
        "method_family": "statistical",
        "description": "ARIMA over an Indian mutual fund's NAV history from AMFI data.",
        "requires": ["fund_name"],
        "returns": "NAV forecast + historical returns",
        "skill": {"metric": "RMSE", "value": "per-scheme"},
        "when_to_use": "the question names a mutual fund scheme rather than a stock",
        "when_not_to_use": "for listed equities -- use the price models",
        "cost": "medium",
        "entry_point": "finance.mutual_funds.forecast_nav",
    },

    # ---------------- astronomy ----------------
    {
        "id": "astronomy.isolation_forest",
        "name": "IsolationForest window detector",
        "domain": "astronomy", "task": DETECT_ANOMALY,
        "method_family": "machine_learning",
        "description": "Slices a light curve into windows, extracts 8 statistical features, "
                       "and isolates outliers. The project's best-performing light-curve detector.",
        "requires": ["target"],
        "returns": "flagged windows with severity, reasons, and a hedged possible cause",
        "skill": {"metric": "ROC-AUC", "value": 0.578,
                  "note": "median over 30 targets against published transit ephemerides. "
                          "Strongly depth-dependent: ~0.51 below 500ppm, 0.90 above 10000ppm "
                          "(r=0.87 with log depth). Shallow transits are not detectable this way."},
        "when_to_use": "the user asks what is unusual in a light curve or names a KIC/TIC id",
        "when_not_to_use": "to confirm a planet -- a flagged dip is a candidate, never a detection",
        "cost": "medium",
        "entry_point": "astronomy.anomaly_detector.run_pipeline",
    },
    {
        "id": "astronomy.autoencoder",
        "name": "1D-CNN autoencoder reconstruction error",
        "domain": "astronomy", "task": DETECT_ANOMALY,
        "method_family": "deep_learning",
        "description": "A small convolutional autoencoder trained on transit-free windows; "
                       "reconstruction error is a second anomaly score, and its 8-dim latent "
                       "vector is the embedding used for RAG retrieval.",
        "requires": ["target"],
        "returns": "reconstruction error per window + embeddings",
        "skill": {"metric": "ROC-AUC", "value": 0.500,
                  "note": "AT CHANCE. Measured once the confound was removed (it previously "
                          "trained on IsolationForest's own output). IsolationForest beats it "
                          "on 91.7% of targets. Kept for its embeddings, NOT for detection."},
        "when_to_use": "only when embeddings are needed for similarity retrieval",
        "when_not_to_use": "for anomaly detection -- it performs at chance; prefer IsolationForest",
        "cost": "slow",
        "entry_point": "astronomy.ts_embedder.train_autoencoder",
    },
    {
        "id": "astronomy.koi_lookup",
        "name": "NASA KOI catalogue lookup",
        "domain": "astronomy", "task": LOOKUP_FACT,
        "method_family": "catalogue",
        "description": "What NASA actually knows about a star: confirmed planets, their "
                       "periods, depths, and disposition. Published fact, not inference.",
        "requires": ["target"],
        "returns": "planet names, periods, depths, CONFIRMED/FALSE POSITIVE",
        "skill": {"metric": "n/a", "value": None, "note": "authoritative catalogue, 9,564 KOIs"},
        "when_to_use": "the user asks whether a star HAS planets, or how many, or their periods -- "
                       "always prefer a recorded fact over a detector's guess",
        "when_not_to_use": "to find NEW anomalies -- this only reports what is already known",
        "cost": "fast",
        "entry_point": "astronomy.koi_catalog.target_summary",
    },
    {
        "id": "astronomy.image_extract",
        "name": "Light-curve extraction from a plot image",
        "domain": "astronomy", "task": EXTRACT_SERIES,
        "method_family": "rule",
        "description": "Traces pixels in an uploaded plot back into a flux series so an image "
                       "can be analysed like data.",
        "requires": ["image"],
        "returns": "a flux series (relative shape only)",
        "skill": {"metric": "round-trip correlation", "value": 0.2,
                  "note": "POOR on dense real light curves (0.19-0.34). Locates dips on clean "
                          "sparse plots but DEPTH IS BROKEN -- normalisation forces the minimum "
                          "to 0, so a 5% and an 8% dip both read as 99.7%."},
        "when_to_use": "the user uploads an image of a light curve",
        "when_not_to_use": "when real data is available -- always prefer fetching the target",
        "cost": "fast",
        "entry_point": "astronomy.image_lightcurve.extract_lightcurve_from_image",
    },

    # ---------------- solar ----------------
    {
        "id": "solar.hawkes_probability",
        "name": "Hawkes self-exciting flare probability",
        "domain": "solar", "task": FORECAST_EVENT_PROBABILITY,
        "method_family": "statistical",
        "description": "A marked Hawkes point process fitted on 1996-2020 GOES data, giving "
                       "P(M-class or larger flare in the next 24h) in closed form. Reports the "
                       "NOAA operational forecast alongside its own number.",
        "requires": [],           # there is only one Sun -- no entity to resolve
        "returns": "probability + NOAA baseline + drivers + caveats",
        "skill": {"metric": "TSS", "value": 0.572,
                  "note": "final block, threshold tuned in-sample -- an optimistic upper bound. "
                          "Branching ratio 0.82, excitation half-life 27h."},
        "when_to_use": "the user asks the chance of a solar flare, or about space weather risk",
        "when_not_to_use": "NEVER to state WHEN a flare will occur -- only a probability over a window",
        "cost": "fast",
        "entry_point": "solar.forecast.current_forecast",
    },
    {
        "id": "solar.hawkes_expected_count",
        "name": "Hawkes expected flare count",
        "domain": "solar", "task": FORECAST_EVENT_COUNT,
        "method_family": "statistical",
        "description": "Expected NUMBER of M+ flares in a window, by Ogata-thinning simulation. "
                       "Distinct from the probability model: simulation is genuinely required "
                       "here because each simulated flare excites further flares, and that "
                       "branching cascade has no closed form.",
        "requires": ["horizon_hours"],
        "returns": "expected event count",
        "skill": {"metric": "TSS", "value": None,
                  "note": "shares the fitted Hawkes parameters; the count itself is not "
                          "separately backtested"},
        "when_to_use": "the user asks HOW MANY flares to expect, not whether one will happen",
        "when_not_to_use": "for a yes/no or probability question -- use the closed-form model",
        "cost": "medium",
        "entry_point": "solar.flare_model.HawkesModel.expected_count",
    },
    {
        "id": "solar.persistence",
        "name": "Persistence baseline",
        "domain": "solar", "task": FORECAST_EVENT_PROBABILITY,
        "method_family": "rule",
        "description": "One line: was there an M+ flare in the last 24 hours? Exists to be "
                       "beaten, and is surprisingly hard to beat.",
        "requires": [],
        "returns": "0 or 1",
        "skill": {"metric": "TSS", "value": 0.43,
                  "note": "0.394-0.464 across walk-forward folds -- BEATS XGBoost in 2 of 4. "
                          "Most short-horizon flare skill is knowing whether the Sun was "
                          "already flaring."},
        "when_to_use": "as a comparison baseline, or when the user asks how hard the problem is",
        "when_not_to_use": "as the headline forecast -- report it beside the model, not instead",
        "cost": "fast",
        "entry_point": "solar.flare_model.persistence_forecast",
    },
    {
        "id": "solar.xgboost_classifier",
        "name": "XGBoost flare classifier",
        "domain": "solar", "task": FORECAST_EVENT_PROBABILITY,
        "method_family": "machine_learning",
        "description": "Gradient-boosted trees over 39 engineered history features. The same "
                       "tool used for price forecasting, applied to a point process -- the "
                       "project's controlled cross-domain comparison.",
        "requires": [],
        "returns": "probability of an M+ flare",
        "skill": {"metric": "TSS", "value": 0.42,
                  "note": "0.193-0.576 across folds, highly variable with solar-cycle phase. "
                          "Loses to the one-line persistence rule in 2 of 4 folds."},
        "when_to_use": "when comparing modelling approaches on the solar domain",
        "when_not_to_use": "as the live forecast -- the Hawkes model is the production path",
        "cost": "slow",
        "entry_point": "solar.flare_model.train_xgboost",
    },
    # ---------------------------------------------------------------- risk
    {
        "id": "finance.risk_profile",
        "name": "Stock risk profile (regime-conditional)",
        "domain": "finance", "task": ASSESS_RISK,
        "method_family": "statistical",
        "description": "How much a single stock could lose, how much of its movement is the "
                       "market versus the company, and which of its unusual days were which. "
                       "Answers the forward-looking question anomaly detection cannot.",
        "requires": ["ticker"],
        "returns": "volatility, downside estimates, drawdown, regime state, attributed anomalies",
        "skill": {"metric": "VaR backtest", "value": "Kupiec p=0.273, Christoffersen p=0.237",
                  "note": "regime-conditional passes both coverage tests; the unconditional "
                          "version is REJECTED (Kupiec p=0.0068 at 95%, p=0.0005 at 99%)"},
        "when_to_use": "the user asks how risky a stock is, or what they could lose",
        "when_not_to_use": "the user asks whether to buy -- the system does not advise",
        "cost": "medium",
        "entry_point": "finance.risk_profile.risk_profile",
    },
    {
        "id": "finance.market_regime",
        "name": "Markov regime + turbulence",
        "domain": "finance", "task": FORECAST_EVENT_PROBABILITY,
        "method_family": "statistical",
        "description": "Two-state hidden Markov model over returns, volatility and "
                       "cross-sectional turbulence. Says whether markets are calm or choppy "
                       "now, and the chance that changes.",
        "requires": [],
        "returns": "regime state, transition probabilities, multi-day risk",
        "skill": {"metric": "lift on event days", "value": 5.59,
                  "note": "P(turbulent | event day) 52.9% vs 9.5% on ordinary days, computed "
                          "causally. Smoothed and filtered states agree on 97.7% of days."},
        "when_to_use": "the user asks about market conditions, volatility or risk generally",
        "when_not_to_use": "the user asks about one stock in isolation",
        "cost": "medium",
        "entry_point": "quant.regimes.turbulent_flag",
    },
    # ------------------------------------------------- anomaly, re-measured
    {
        "id": "finance.hampel_anomaly",
        "name": "Robust threshold detector",
        "domain": "finance", "task": DETECT_ANOMALY,
        "method_family": "statistical",
        "description": "Flags days far from the recent normal, measured with median and MAD "
                       "so that one extreme day cannot hide itself by inflating the spread.",
        "requires": ["ticker"],
        "returns": "flagged days + reasons",
        "skill": {"metric": "out-of-sample lift", "value": 2.84,
                  "note": "beats chance in 17 of 18 independent held-out evaluations -- the "
                          "most RELIABLE detector tested, though not the highest peak. "
                          "One-Class SVM scored 4.62x in-sample and 1.07x held out."},
        "when_to_use": "the user asks which days were unusual for a stock",
        "when_not_to_use": "the series is periodic -- use a period search instead",
        "cost": "fast",
        "entry_point": "anomaly.detectors.HampelMAD",
    },
    {
        "id": "astronomy.bls_transit",
        "name": "Box Least Squares period search",
        "domain": "astronomy", "task": DETECT_ANOMALY,
        "method_family": "statistical",
        "description": "Searches for a repeating box-shaped dip, folding the light curve on "
                       "thousands of candidate periods. Finds transits that no window-level "
                       "detector can, because it stacks every repetition.",
        "requires": ["kepid"],
        "returns": "period, epoch, depth, detection significance, folded curve",
        "skill": {"metric": "period recovery within 1%", "value": 0.68,
                  "note": "68% of 25 targets recovered, median significance 29.2. Failures "
                          "are mostly harmonic locks (finding P/3 or 2P). Generic anomaly "
                          "detectors reach only 1.2-1.4x chance on the same data."},
        "when_to_use": "a light curve, and the question is whether a planet is there",
        "when_not_to_use": "the series is not periodic",
        "cost": "slow",
        "entry_point": "astronomy.transit_search.search",
    },
    {
        "id": "shared.structure_profiler",
        "name": "Series structure profiler",
        "domain": "any", "task": LOOKUP_FACT,
        "method_family": "statistical",
        "description": "Inspects an unfamiliar series and decides whether its events REPEAT "
                       "or happen once, which determines the whole detector family. Uses no "
                       "labels and no domain tag.",
        "requires": ["series"],
        "returns": "structure (periodic / one-off / uncertain) + recommended family",
        "skill": {"metric": "classification accuracy", "value": 1.0,
                  "note": "10/10 on held-out series. The decision matters: scored the wrong "
                          "way round, the matrix profile gives ROC 0.303 on a Kepler target; "
                          "the right way round, 0.697."},
        "when_to_use": "before choosing a detector for a series never seen before",
        "when_not_to_use": "the structure is already known",
        "cost": "fast",
        "entry_point": "anomaly.profile.profile_and_recommend",
    },
]


# --------------------------------------------------------------------------
# lookups
# --------------------------------------------------------------------------

def by_id(model_id: str) -> ModelSpec | None:
    return next((m for m in REGISTRY if m["id"] == model_id), None)


def feasible(domain: str | None = None, task: str | None = None,
             available: set | None = None) -> list[ModelSpec]:
    """The capability filter: models that CAN answer, given what is known.

    This is the deterministic half of selection. Anything it removes is
    removed on hard grounds -- wrong domain, wrong task, missing prerequisite
    -- so the LLM downstream cannot choose something impossible however
    confidently it argues for it. Small local models are unreliable at
    structured choice (documented throughout this project), and constraining
    the option set is more dependable than asking them to be careful.
    """
    available = available or set()
    out = []
    for m in REGISTRY:
        if domain and m["domain"] not in (domain, "any"):
            continue
        if task and m["task"] != task:
            continue
        if any(r not in available for r in m.get("requires", [])):
            continue
        out.append(m)
    return out


def describe_for_prompt(models: list[ModelSpec]) -> str:
    """Render a shortlist for the LLM, skill and limits included.

    The limits are not padding: the autoencoder's entry says it performs at
    chance, so a model reading this list has what it needs to avoid choosing
    it -- and, if it does, the answer can say so.
    """
    lines = []
    for m in models:
        skill = m.get("skill", {})
        value = skill.get("value")
        metric = skill.get("metric", "")
        score = f"{metric} {value}" if value is not None else "not measured"
        lines.append(
            f"- id: {m['id']}\n"
            f"    name       : {m['name']} ({m['method_family']})\n"
            f"    task       : {m['task']}\n"
            f"    does       : {m['description']}\n"
            f"    measured   : {score}"
            + (f" -- {skill['note']}" if skill.get("note") else "") + "\n"
            f"    use when   : {m['when_to_use']}\n"
            f"    avoid when : {m['when_not_to_use']}"
        )
    return "\n".join(lines)


def registry_stats() -> dict:
    from collections import Counter
    return {
        "n_models": len(REGISTRY),
        "by_domain": dict(Counter(m["domain"] for m in REGISTRY)),
        "by_task": dict(Counter(m["task"] for m in REGISTRY)),
        "by_method": dict(Counter(m["method_family"] for m in REGISTRY)),
        "measured": sum(1 for m in REGISTRY if m.get("skill", {}).get("value") is not None),
    }


if __name__ == "__main__":
    import json

    stats = registry_stats()
    print("Model registry\n" + "=" * 62)
    print(json.dumps(stats, indent=2))

    print("\nTasks that cut across domains:")
    for task in TASKS:
        doms = sorted({m["domain"] for m in REGISTRY if m["task"] == task})
        if doms:
            print(f"  {task:<24}{doms}")

    print("\nExample capability filter -- a solar question, nothing else known:")
    for m in feasible(domain="solar"):
        print(f"  {m['id']:<34}{m['task']}")

    print("\nExample -- 'is this light curve unusual?' (astronomy, detect_anomaly):")
    print(describe_for_prompt(feasible(domain="astronomy", task=DETECT_ANOMALY,
                                       available={"target"})))
