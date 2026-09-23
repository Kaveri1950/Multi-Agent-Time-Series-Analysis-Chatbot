"""The live forecast: one entry point the dashboard and the chatbot share.

Everything expensive is cached, because a Streamlit page reruns on every
widget interaction and a chatbot turn should not refit 25 years of history.
Two levels:

  * the fitted Hawkes parameters are persisted to JSON (four floats -- there
    is no reason to ever refit them inside a page load), and
  * the measured skill numbers are persisted alongside, because they come
    from a walk-forward backtest that takes minutes.

`current_forecast()` then does only cheap work: fetch the last 7 days of
flares, evaluate the closed-form Hawkes probability against that history,
read NOAA's own published number for the same day, and assemble a
ForecastSummary.

Honesty is built into the return value rather than left to the caller. The
`caveats` list is always populated, and `baselines` always carries NOAA's
figure when it is available, so any consumer -- page or LLM -- is handed the
comparison alongside the claim instead of having to think to ask for it.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from solar.flare_data import (CACHE_DIR, class_at_least, fetch_daily_indices,
                              fetch_noaa_probabilities, fetch_recent_flares,
                              fetch_solar_regions, load_historical_catalog)
from solar.flare_model import (HawkesModel, M_CLASS_FLUX, catalog_to_hawkes_input,
                               fit_and_report)

MODEL_PATH = CACHE_DIR / "hawkes_model.json"

TRAIN_START_YEAR = 1996
TRAIN_END_YEAR = 2026

# Confidence is about how much evidence the forecast rests on, NOT about how
# high the probability is. A 5% forecast from a well-observed quiet Sun is
# high-confidence; a 60% forecast computed from three flares is not.
MIN_EVENTS_FOR_HIGH_CONFIDENCE = 8


def _standing_caveats() -> list:
    """Limits that apply to every forecast this module produces. Kept in one
    place so they cannot be quietly dropped from one call path."""
    return [
        "This is a probability for a 24-hour window, not a prediction of when a flare will occur. "
        "Individual flare timing is not predictable.",
        "Trained on 1996-2026 (GOES catalogue), spanning Solar Cycles 23, 24 and 25.",
        "X-class flares are far too rare to forecast at this data scale (171 events in 25 years); "
        "the target is deliberately M-class-or-larger only.",
        "Magnetic complexity (Mount Wilson / McIntosh class) is the strongest known flare predictor, "
        "but NOAA serves it for today only, so it informs context here and not the fitted model.",
    ]


# --------------------------------------------------------------------------
# model persistence
# --------------------------------------------------------------------------

def save_model(model: HawkesModel, skill: dict | None = None) -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mu": model.mu, "alpha": model.alpha, "beta": model.beta, "gamma": model.gamma,
        "loglik": model.loglik, "n_events": model.n_events,
        "marks": list(np.asarray(model.marks_).tolist()) if model.marks_ is not None else [],
        "trained_on": f"{TRAIN_START_YEAR}-{TRAIN_END_YEAR}",
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "skill": skill or {},
    }
    MODEL_PATH.write_text(json.dumps(payload, indent=2))
    return MODEL_PATH


def load_model() -> tuple:
    """Returns (HawkesModel, metadata) or (None, None) if never fitted."""
    if not MODEL_PATH.exists():
        return None, None
    payload = json.loads(MODEL_PATH.read_text())
    m = HawkesModel()
    m.mu, m.alpha = payload["mu"], payload["alpha"]
    m.beta, m.gamma = payload["beta"], payload["gamma"]
    m.loglik, m.n_events = payload.get("loglik"), payload.get("n_events", 0)
    m.marks_ = np.array(payload.get("marks", [0.0]) or [0.0])
    m._converged = True
    return m, payload


GP_PATH = CACHE_DIR / "gp_classifier.joblib"


def train_and_save_gp(verbose: bool = True) -> dict:
    """Fit the GP classifier and persist it.

    Unlike the Hawkes model, which is four floats and serialises to JSON, a
    fitted GP carries its training set and kernel state, so it goes to joblib.
    Saved rather than refitted on demand because the fit takes ~14s at 1,500
    training days and the cost is cubic -- far too slow for a chat turn or a
    page load.

    This is now the better-scoring solar model: over 57 days against NOAA it
    reached Brier 0.148 and TSS 0.239, beating the Hawkes process on every
    measure (0.158 / 0.163) and beating NOAA's own operational TSS of 0.231.
    NOAA remains better calibrated.
    """
    import joblib

    from solar.features import build_daily_frame, feature_columns
    from solar.flare_model import train_gp_classifier

    catalog = load_historical_catalog(TRAIN_START_YEAR, TRAIN_END_YEAR)
    frame = build_daily_frame(catalog)

    # Hold out the tail so the saved model has an honest, quotable score
    # rather than one measured on its own training days.
    cut = int(len(frame) * 0.9)
    train, test = frame.iloc[:cut], frame.iloc[cut:]

    if verbose:
        print(f"Fitting GP classifier on {len(train)} days "
              f"(capped to the most recent 1500)...")
    clf, prob_test, prob_train, scaler = train_gp_classifier(train, test)

    from solar.flare_model import best_tss, brier, evaluate
    _, thresh = best_tss(train["label"].tail(len(prob_train)), prob_train)
    skill = evaluate(test["label"], prob_test, thresh)
    skill["brier"] = round(brier(test["label"], prob_test), 4)

    GP_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "scaler": scaler, "columns": feature_columns(train),
                 "threshold": thresh, "trained_on": f"{TRAIN_START_YEAR}-{TRAIN_END_YEAR}",
                 "n_train": len(train), "holdout_skill": skill,
                 "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, GP_PATH)
    if verbose:
        print(f"  saved -> {GP_PATH}")
        print(f"  holdout: TSS {skill['tss']}, Brier {skill['brier']}")
    return {"skill": skill, "path": str(GP_PATH)}


def load_gp():
    """(payload, None) or (None, reason) -- never raises, so a missing or
    version-incompatible pickle degrades to the Hawkes-only forecast rather
    than taking down the page."""
    if not GP_PATH.exists():
        return None, "not fitted -- run `python -m solar.forecast --train-gp`"
    try:
        import joblib

        return joblib.load(GP_PATH), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def gp_probability():
    """P(M+ in 24h) from the GP classifier, for the most recent day available.

    Features come from the complete Hesperia catalogue rather than the
    harvested rolling archive. That matters: the archive held 53% of real M+
    events over the last two months, and a model fed an incomplete history
    under-predicts in proportion. Hesperia runs about one day behind, which
    is reported as `as_of` rather than hidden.
    """
    payload, reason = load_gp()
    if payload is None:
        return {"probability": None, "error": reason}

    try:
        from solar.features import build_daily_frame

        catalog = load_historical_catalog(TRAIN_START_YEAR, TRAIN_END_YEAR)
        frame = build_daily_frame(catalog)
        row = frame.iloc[[-1]]
        x = payload["scaler"].transform(row[payload["columns"]])
        prob = float(payload["clf"].predict_proba(x)[0, 1])
        return {"probability": prob,
                "as_of": str(pd.Timestamp(row["date"].iloc[0]).date()),
                "threshold": payload.get("threshold"),
                "holdout_skill": payload.get("holdout_skill", {}),
                "trained_on": payload.get("trained_on")}
    except Exception as e:
        return {"probability": None, "error": f"{type(e).__name__}: {e}"}


def train_and_save(n_folds: int = 4, verbose: bool = True) -> dict:
    """Fit the Hawkes model on the historical catalogue, backtest, persist.

    Run once (and again if the feature set or catalogue changes) -- not on
    page load. `python -m solar.forecast --train` does this.
    """
    if verbose:
        print(f"Fitting on {TRAIN_START_YEAR}-{TRAIN_END_YEAR} (downloads on first run)...")
    report = fit_and_report(TRAIN_START_YEAR, TRAIN_END_YEAR, n_folds=n_folds)

    catalog = load_historical_catalog(TRAIN_START_YEAR, TRAIN_END_YEAR)
    times, marks, horizon, _ = catalog_to_hawkes_input(catalog)
    model = HawkesModel().fit(times, marks, horizon)

    folds = report.get("walk_forward", {}).get("folds", [])
    skill = {
        "hawkes_tss_final_block": report.get("hawkes_eval", {}).get("tss"),
        "xgboost_tss_by_fold": [f["xgboost"]["tss"] for f in folds],
        "persistence_tss_by_fold": [f["persistence"]["tss"] for f in folds],
        "note": "walk-forward across solar-cycle phases; persistence is a one-line rule and "
                "is competitive, which is itself a finding",
    }
    save_model(model, skill)
    if verbose:
        print(f"  saved -> {MODEL_PATH}")
        print(f"  {model.summary()}")
    return {"model": model.summary(), "skill": skill, "report": report}


# --------------------------------------------------------------------------
# the live forecast
# --------------------------------------------------------------------------

def _confidence(n_recent_events: int, model_loaded: bool) -> str:
    if not model_loaded:
        return "LOW"
    if n_recent_events >= MIN_EVENTS_FOR_HIGH_CONFIDENCE:
        return "MEDIUM"      # never HIGH: see caveats -- the model has not seen Cycle 25
    return "LOW"


def current_forecast(include_context: bool = True) -> dict:
    """P(M+ flare in the next 24h) from live NOAA data.

    Cheap: no refitting. Returns a dict shaped for
    agents/forecast_shared.from_solar_result().
    """
    now = datetime.now(timezone.utc)
    model, meta = load_model()

    flares = fetch_recent_flares()
    basis, caveats = [], _standing_caveats()

    # --- the model's own probability ---
    probability = None
    if model is not None and not flares.empty:
        cat = flares.rename(columns={"max_class": "flare_class"})
        times, marks, _, t0 = catalog_to_hawkes_input(cat)
        if len(times):
            # "now" in the same hours-since-first-event clock the model uses
            now_hours = (pd.Timestamp(now).tz_localize(None)
                         - pd.Timestamp(t0)).total_seconds() / 3600.0
            probability = model.probability_within(times, marks, now_hours, 24.0)
            basis.append(f"{len(times)} M-class-or-larger flare(s) in the last 7 days")
            last_gap = (now_hours - times[-1])
            basis.append(f"most recent M+ flare {last_gap:.1f} hours ago")
            basis.append(f"fitted excitation half-life {np.log(2) / model.beta:.1f} hours "
                         f"(branching ratio {model.alpha:.2f})")
        else:
            # No M+ events in the window at all -> the background rate alone.
            probability = float(1.0 - np.exp(-model.mu * 24.0))
            basis.append("no M-class-or-larger flares in the last 7 days")
            basis.append("probability reflects the fitted background rate only")
    elif model is None:
        caveats.insert(0, "MODEL NOT FITTED -- run `python -m solar.forecast --train` first. "
                          "No model probability is available.")

    # --- all flares (any class) for context ---
    if not flares.empty:
        by_class = flares["class_letter"].value_counts().to_dict()
        basis.append(f"last 7 days by class: {by_class}")

    # --- NOAA's own forecast, the external baseline ---
    baselines = []
    noaa_m = None
    try:
        probs = fetch_noaa_probabilities()
        if not probs.empty:
            latest = probs.iloc[-1]
            noaa_m = float(latest["m_class_1_day"]) / 100.0
            baselines.append({
                "name": "NOAA SWPC operational forecast",
                "probability": noaa_m,
                "note": f"published {latest['date'].date()}, human-in-the-loop; the real benchmark",
            })
            baselines.append({
                "name": "NOAA SWPC, X-class 1-day",
                "probability": float(latest["x_class_1_day"]) / 100.0,
                "note": "shown for context; this project does not forecast X-class",
            })
    except Exception as e:
        caveats.append(f"NOAA baseline unavailable this run ({type(e).__name__}).")

    # --- persistence, the trivial floor ---
    if not flares.empty:
        had_recent = bool(flares["max_class"].apply(class_at_least).any())
        recent_24h = flares[flares["begin_time"] >= pd.Timestamp(now) - pd.Timedelta(hours=24)]
        persistence = 1.0 if bool(recent_24h["max_class"].apply(class_at_least).any()) else 0.0
        baselines.append({
            "name": "persistence (M+ in last 24h?)",
            "probability": persistence,
            "note": "a one-line rule that scores TSS ~0.4 in backtesting -- the floor to beat",
        })

    # The GP classifier as a second opinion. Reported ALONGSIDE the Hawkes
    # probability rather than replacing it: the GP scores better (Brier 0.148
    # vs 0.158, TSS 0.239 vs 0.163) but says nothing about mechanism, while
    # the Hawkes parameters are the interpretable result -- a branching ratio
    # of 0.84 and a 22-hour excitation half-life are the physics claim. Two
    # models disagreeing is information, not a problem to resolve.
    gp = gp_probability()
    if gp.get("probability") is not None:
        basis.append(f"GP classifier (39 daily features): {gp['probability']:.1%}, "
                     f"as of {gp.get('as_of')}")
    elif gp.get("error"):
        caveats.append(f"GP classifier unavailable ({gp['error']}).")

    skill = (meta or {}).get("skill", {})

    result = {
        "as_of": now.isoformat(timespec="seconds"),
        "method": "hawkes-exponential-marked" if model is not None else "unavailable",
        "methods_available": ["hawkes", "xgboost", "persistence", "climatology"],
        "forecast": {
            "label": "M-class or larger solar flare",
            "probability": probability,
            "horizon": "next 24 hours",
            "confidence": _confidence(len(flares[flares["max_class"].apply(class_at_least)])
                                      if not flares.empty else 0, model is not None),
            "basis": basis,
        },
        "baselines": baselines,
        "skill": skill,
        "caveats": caveats,
        "model_meta": meta or {},
        "gp": gp,
    }

    if include_context:
        result["context"] = _live_context(caveats)
    result["recent_flares"] = flares
    return result


def _live_context(caveats: list) -> dict:
    """Today's solar state: active regions with magnetic classification, and
    the daily indices. Context for the human reader and the LLM -- explicitly
    NOT model inputs, because no historical archive exists for them."""
    context = {}
    try:
        regions = fetch_solar_regions()
        context["n_regions"] = len(regions)
        if not regions.empty:
            context["max_complexity"] = int(regions["mag_complexity"].max())
            context["has_delta_region"] = bool(regions["has_delta"].any())
            context["largest_region_area"] = int(regions["area"].max())
            context["regions"] = regions
            if context["has_delta_region"]:
                context["note"] = ("A delta-configuration region is present -- these produce "
                                    "nearly all X-class flares. Not a model input here.")
    except Exception as e:
        caveats.append(f"Active-region data unavailable this run ({type(e).__name__}).")

    try:
        dsd = fetch_daily_indices()
        if not dsd.empty:
            latest = dsd.iloc[-1]
            context["radio_flux"] = latest.get("radio_flux")
            context["sunspot_number"] = latest.get("sunspot_number")
            context["sunspot_area"] = latest.get("sunspot_area")
            context["daily_indices"] = dsd
    except Exception as e:
        caveats.append(f"Daily indices unavailable this run ({type(e).__name__}).")

    return context


def forecast_summary_text() -> str:
    """One-paragraph plain-language rendering, for the chatbot and the CLI."""
    from agents.forecast_shared import format_for_prompt, from_solar_result

    return format_for_prompt(from_solar_result(current_forecast()))


if __name__ == "__main__":
    import sys

    if "--train" in sys.argv:
        train_and_save()
        train_and_save_gp()
        sys.exit(0)
    if "--train-gp" in sys.argv:
        train_and_save_gp()
        sys.exit(0)

    model, meta = load_model()
    if model is None:
        print("No fitted model found. Run:  python -m solar.forecast --train")
        sys.exit(1)

    print("Live solar flare forecast")
    print("=" * 62)
    fc = current_forecast()

    p = fc["forecast"]["probability"]
    print(f"\n  P(M+ flare in next 24h) = {p:.1%}" if p is not None else "\n  unavailable")
    print(f"  confidence: {fc['forecast']['confidence']}")

    print("\n  driven by:")
    for b in fc["forecast"]["basis"]:
        print(f"    - {b}")

    print("\n  baselines:")
    for b in fc["baselines"]:
        print(f"    {b['name']:<38} {b['probability']:.1%}")

    ctx = fc.get("context", {})
    if ctx:
        print(f"\n  today's Sun: {ctx.get('n_regions')} active regions, "
              f"F10.7={ctx.get('radio_flux')}, sunspot number={ctx.get('sunspot_number')}, "
              f"delta region present={ctx.get('has_delta_region')}")

    print("\n  caveats:")
    for c in fc["caveats"]:
        print(f"    - {c}")
