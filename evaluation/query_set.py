"""Labelled queries for measuring model selection.

WHAT THIS IS FOR: agents/model_selector.py can choose between 14 models via
three strategies (deterministic capability filter, unconstrained LLM, and the
hybrid that constrains the LLM to feasible options). Which is best is an
empirical question, and answering it needs queries with known correct
answers. A pilot of 8 demonstrated the mechanism; it did not measure it.

HOW THE LABELS WERE DECIDED. `expected` is the model that SHOULD answer,
judged by the question actually asked rather than by keyword overlap:

  * "what will the price be"        -> a forecast        (arima_forecast)
  * "how accurate is the forecast"  -> an evaluation     (arima_backtest)
  * "does sentiment help"           -> a comparison      (xgboost_sentiment)

All three mention price and models, and a keyword matcher confuses them. The
distinctions are the point of the task taxonomy, so the set deliberately
contains near-identical phrasings that resolve differently.

NOTE ON LABEL DRIFT: value-forecast queries expect `finance.gpr_logged_forecast`, not the
ARIMA entry. The chatbot's forecast path was switched to the Gaussian Process, and a label set
that still named the old model would score the system wrong for being right. Labels track the
production path; when that moves, these move with it.

`expected = None` marks questions where NO model should run -- definitional
or conceptual questions answerable from general knowledge. A system that
fires a pipeline at "what does RSI mean" is wasting minutes on a dictionary
lookup, so not selecting is the correct behaviour and is scored as such.

CATEGORIES, so results can be broken down rather than averaged into mush:
  canonical   -- the plainest phrasing of an intent
  paraphrase  -- the same intent said differently, incl. terse and clumsy
  adversarial -- the naive reading is wrong (cross-domain traps, near-misses)
  general     -- no model should be selected

KNOWN LIMITATION, stated because it affects how the numbers should be read:
these queries were written by the same author as the selector's patterns, so
accuracy here is an upper bound. Phrasings contributed by someone who has not
seen the regexes would be a stronger test, and the set is structured so
adding them means appending rows.
"""

CANONICAL, PARAPHRASE, ADVERSARIAL, GENERAL = "canonical", "paraphrase", "adversarial", "general"

FIN = {"ticker"}
FIN_H = {"ticker", "horizon_days"}
AST = {"target"}
SOL: set = set()
SOL_H = {"horizon_hours"}


def q(query, domain, available, expected, category=PARAPHRASE):
    return {"query": query, "domain": domain, "available": available,
            "expected": expected, "category": category}


QUERIES = [

    # ================= FINANCE : forecast a value =================
    q("what will RELIANCE close at in 30 days", "finance", FIN_H, "finance.gpr_logged_forecast", CANONICAL),
    q("predict the price of TCS for the next month", "finance", FIN_H, "finance.gpr_logged_forecast"),
    q("where is INFY headed over the next 60 days", "finance", FIN_H, "finance.gpr_logged_forecast"),
    q("forecast HDFCBANK share price", "finance", FIN_H, "finance.gpr_logged_forecast"),
    q("what's the price target for WIPRO in two weeks", "finance", FIN_H, "finance.gpr_logged_forecast"),
    q("project ITC forward ninety days", "finance", FIN_H, "finance.gpr_logged_forecast"),
    q("give me a price prediction for SBIN", "finance", FIN_H, "finance.gpr_logged_forecast"),
    q("how much will TATAMOTORS be worth next month", "finance", FIN_H, "finance.gpr_logged_forecast"),
    q("future price of ONGC please", "finance", FIN_H, "finance.gpr_logged_forecast"),
    q("estimate RELIANCE price 45 days out", "finance", FIN_H, "finance.gpr_logged_forecast"),

    # ================= FINANCE : evaluate the model =================
    q("how accurate is your price model", "finance", FIN, "finance.arima_backtest", CANONICAL),
    q("how reliable are these forecasts", "finance", FIN, "finance.arima_backtest"),
    q("what's the RMSE on TCS", "finance", FIN, "finance.arima_backtest"),
    q("how well does the model actually do", "finance", FIN, "finance.arima_backtest"),
    q("can I trust this prediction", "finance", FIN, "finance.arima_backtest"),
    q("what's your error rate for INFY", "finance", FIN, "finance.arima_backtest"),
    q("how good is the ARIMA fit", "finance", FIN, "finance.arima_backtest"),
    q("show me the backtest results", "finance", FIN, "finance.arima_backtest"),
    # adversarial: mentions "predict" but asks about accuracy, not a forecast
    q("how accurate were your past predictions for RELIANCE", "finance", FIN,
      "finance.arima_backtest", ADVERSARIAL),
    q("how much error does the price forecast have", "finance", FIN,
      "finance.arima_backtest", ADVERSARIAL),

    # ================= FINANCE : compare models =================
    q("does news sentiment actually help prediction", "finance", FIN,
      "finance.xgboost_sentiment", CANONICAL),
    q("is the model better with or without sentiment", "finance", FIN, "finance.xgboost_sentiment"),
    q("compare the models with and without news", "finance", FIN, "finance.xgboost_sentiment"),
    q("does adding sentiment improve RMSE", "finance", FIN, "finance.xgboost_sentiment"),
    q("is XGBoost better than ARIMA here", "finance", FIN, "finance.xgboost_sentiment"),
    q("what difference does sentiment make to accuracy", "finance", FIN, "finance.xgboost_sentiment"),
    # adversarial: says "sentiment" but wants the comparison, not the scores
    q("does sentiment matter for forecasting TCS", "finance", FIN,
      "finance.xgboost_sentiment", ADVERSARIAL),

    # ================= FINANCE : detect anomalies =================
    q("any anomalies in RELIANCE", "finance", FIN, "finance.anomaly_detectors", CANONICAL),
    q("anything unusual about TCS recently", "finance", FIN, "finance.anomaly_detectors"),
    q("were there outliers in INFY", "finance", FIN, "finance.anomaly_detectors"),
    q("show me suspicious trading days for HDFCBANK", "finance", FIN, "finance.anomaly_detectors"),
    q("what was weird about SBIN this year", "finance", FIN, "finance.anomaly_detectors"),
    q("find irregular days in WIPRO", "finance", FIN, "finance.anomaly_detectors"),
    q("any abnormal price movement in ITC", "finance", FIN, "finance.anomaly_detectors"),
    q("which days were strange for TATAMOTORS", "finance", FIN, "finance.anomaly_detectors"),
    q("detect outliers for ONGC", "finance", FIN, "finance.anomaly_detectors"),
    q("spot anything odd in RELIANCE volume", "finance", FIN, "finance.anomaly_detectors"),
    q("unusual volatility in the Nifty", "finance", FIN, "finance.anomaly_detectors"),

    # ================= FINANCE : sentiment =================
    q("what's the news sentiment on RELIANCE", "finance", FIN,
      "finance.finbert_sentiment", CANONICAL),
    q("how is the market feeling about TCS", "finance", FIN, "finance.finbert_sentiment"),
    q("show me recent headlines for INFY", "finance", FIN, "finance.finbert_sentiment"),
    q("is the news positive or negative for HDFCBANK", "finance", FIN, "finance.finbert_sentiment"),
    q("what are people saying about SBIN", "finance", FIN, "finance.finbert_sentiment"),
    q("sentiment score for WIPRO", "finance", FIN, "finance.finbert_sentiment"),

    # ================= SOLAR : event probability =================
    q("what is the chance of a solar flare tomorrow", "solar", SOL,
      "solar.hawkes_probability", CANONICAL),
    q("will there be a solar flare in the next 24 hours", "solar", SOL, "solar.hawkes_probability"),
    q("how likely is an M-class flare today", "solar", SOL, "solar.hawkes_probability"),
    q("what's the probability of solar activity tomorrow", "solar", SOL, "solar.hawkes_probability"),
    q("odds of a flare today", "solar", SOL, "solar.hawkes_probability"),
    q("risk of a geomagnetic storm", "solar", SOL, "solar.hawkes_probability"),
    q("should I expect space weather disruption tomorrow", "solar", SOL, "solar.hawkes_probability"),
    q("chance of an X-class event", "solar", SOL, "solar.hawkes_probability"),
    q("how likely is a coronal mass ejection", "solar", SOL, "solar.hawkes_probability"),
    q("give me the flare forecast", "solar", SOL, "solar.hawkes_probability"),

    # ================= SOLAR : event count =================
    q("how many solar flares should I expect this week", "solar", SOL_H,
      "solar.hawkes_expected_count", CANONICAL),
    q("how many flares in the next 3 days", "solar", SOL_H, "solar.hawkes_expected_count"),
    q("what number of flares are likely this month", "solar", SOL_H, "solar.hawkes_expected_count"),
    q("count of expected solar events this week", "solar", SOL_H, "solar.hawkes_expected_count"),
    q("how many M-class flares will there be", "solar", SOL_H, "solar.hawkes_expected_count"),
    # adversarial: "how many" + "chance" in one sentence -- count must win
    q("how many flares should I expect, and what are the chances", "solar", SOL_H,
      "solar.hawkes_expected_count", ADVERSARIAL),

    # ================= ASTRONOMY : detect anomalies =================
    q("anything unusual in KIC 5358624", "astronomy", AST,
      "astronomy.isolation_forest", CANONICAL),
    q("any anomalies in KIC 10619192", "astronomy", AST, "astronomy.isolation_forest"),
    q("are there outliers in the light curve of KIC 8544996", "astronomy", AST,
      "astronomy.isolation_forest"),
    q("find irregular windows in KIC 11359879", "astronomy", AST, "astronomy.isolation_forest"),
    q("what looks strange in KIC 6922244", "astronomy", AST, "astronomy.isolation_forest"),
    q("detect anomalies for KIC 8191672", "astronomy", AST, "astronomy.isolation_forest"),
    q("is there anything abnormal in this light curve", "astronomy", AST,
      "astronomy.isolation_forest"),
    q("show me suspicious dips in KIC 5358624", "astronomy", AST, "astronomy.isolation_forest"),

    # ================= ASTRONOMY : catalogue lookup =================
    q("does KIC 5358624 have any confirmed planets", "astronomy", AST,
      "astronomy.koi_lookup", CANONICAL),
    q("how many planets does KIC 10619192 have", "astronomy", AST, "astronomy.koi_lookup"),
    q("what is the orbital period of KIC 8544996", "astronomy", AST, "astronomy.koi_lookup"),
    q("is KIC 10079481 a false positive", "astronomy", AST, "astronomy.koi_lookup"),
    q("what planets are known around KIC 11359879", "astronomy", AST, "astronomy.koi_lookup"),
    q("has KIC 6922244 got a confirmed planet", "astronomy", AST, "astronomy.koi_lookup"),
    q("tell me the known planets for KIC 8191672", "astronomy", AST, "astronomy.koi_lookup"),
    # adversarial: a catalogue question that sounds like detection
    q("did anyone find a planet at KIC 5358624", "astronomy", AST,
      "astronomy.koi_lookup", ADVERSARIAL),

    # ================= FINANCE : assess risk =================
    # Distinct from detect_anomaly, which looks BACKWARDS at which days were
    # odd, and from forecast_event_prob, which asks whether an EVENT will
    # occur. These ask how much could be lost holding the thing. The last two
    # are adversarial: both carry vocabulary the probability pattern claims
    # ("chances", "safe"), and must still land on the risk model.
    q("how risky is TCS", "finance", FIN, "finance.risk_profile", CANONICAL),
    q("how risky is it to hold RELIANCE", "finance", FIN, "finance.risk_profile"),
    q("what could I lose holding INFY", "finance", FIN, "finance.risk_profile"),
    q("how much could I lose on HDFCBANK", "finance", FIN, "finance.risk_profile"),
    q("show me the downside for SBIN", "finance", FIN, "finance.risk_profile"),
    q("what is the maximum drawdown on WIPRO", "finance", FIN, "finance.risk_profile"),
    q("value at risk for ITC", "finance", FIN, "finance.risk_profile"),
    q("give me a risk profile for RELIANCE", "finance", FIN, "finance.risk_profile"),
    q("is RELIANCE a safe stock to hold", "finance", FIN,
      "finance.risk_profile", ADVERSARIAL),
    q("what are the chances I lose money on TCS", "finance", FIN,
      "finance.risk_profile", ADVERSARIAL),

    # ================= FINANCE : forecast volatility =================
    # Distinct from forecast_value, which wants a PRICE, and from
    # assess_risk, which looks at what could be lost holding a position.
    # These ask how big the moves will be. The last two are adversarial:
    # "unusual volatility" belongs to detect_anomaly and "what is volatility"
    # is definitional, so the pattern has to want a forward-looking cue
    # rather than the bare word.
    q("how volatile will TCS be tomorrow", "finance", FIN,
      "finance.har_volatility", CANONICAL),
    q("forecast the volatility of RELIANCE", "finance", FIN, "finance.har_volatility"),
    q("volatility forecast for INFY", "finance", FIN, "finance.har_volatility"),
    q("what is the expected volatility of HDFCBANK", "finance", FIN,
      "finance.har_volatility"),
    q("predict volatility for SBIN", "finance", FIN, "finance.har_volatility"),
    q("how volatile is WIPRO going to be", "finance", FIN, "finance.har_volatility"),
    q("give me the realized volatility outlook for ITC", "finance", FIN,
      "finance.har_volatility"),
    q("how big are the moves going to be in TCS", "finance", FIN,
      "finance.har_volatility", ADVERSARIAL),

    # ================= GENERAL : nothing should run =================
    q("what does RSI mean", "finance", FIN, None, GENERAL),
    q("what is a moving average", "finance", FIN, None, GENERAL),
    q("explain what volatility is", "finance", FIN, None, GENERAL),
    q("what is MACD", "finance", FIN, None, GENERAL),
    q("what is a light curve", "astronomy", AST, None, GENERAL),
    q("how does the transit method work", "astronomy", AST, None, GENERAL),
    q("what is a solar flare", "solar", SOL, None, GENERAL),
    q("why do flares matter for satellites", "solar", SOL, None, GENERAL),
    q("what does M-class mean", "solar", SOL, None, GENERAL),
    q("what is an exoplanet", "astronomy", AST, None, GENERAL),
]


def _expand_paraphrases():
    """Terse and clumsy variants of the canonical queries.

    Real users do not type full sentences. Dropping articles, punctuation and
    politeness is where a pattern matcher is most likely to fail, so those
    variants belong in the set rather than only well-formed English.
    """
    extra = []
    terse = {
        "finance.gpr_logged_forecast": ["RELIANCE 30 day forecast", "TCS price next month?",
                                   "predict INFY", "forecast SBIN 60 days"],
        "finance.arima_backtest": ["RELIANCE model accuracy", "how accurate?",
                                   "TCS rmse", "backtest INFY"],
        "finance.anomaly_detectors": ["RELIANCE anomalies", "TCS outliers",
                                      "anything odd INFY", "weird days SBIN"],
        "finance.finbert_sentiment": ["RELIANCE news", "TCS sentiment",
                                      "INFY headlines"],
        "solar.hawkes_probability": ["flare chance", "solar flare probability",
                                     "flare risk today", "space weather odds"],
        "solar.hawkes_expected_count": ["how many flares this week",
                                        "number of flares next 5 days"],
        "astronomy.isolation_forest": ["KIC 5358624 anomalies", "outliers KIC 10619192",
                                       "anything unusual KIC 8544996"],
        "astronomy.koi_lookup": ["KIC 5358624 planets?", "confirmed planets KIC 10619192",
                                 "KIC 8544996 orbital period"],
    }
    domains = {"finance": FIN_H, "solar": SOL_H, "astronomy": AST}
    for model_id, variants in terse.items():
        domain = model_id.split(".")[0]
        for v in variants:
            extra.append(q(v, domain, domains[domain], model_id, PARAPHRASE))
    return extra


QUERIES.extend(_expand_paraphrases())


# --------------------------------------------------------------------------
# cross-domain traps -- where the naive reading sends it to the wrong domain
# --------------------------------------------------------------------------

ADVERSARIAL_CASES = [
    # "the sun" plus a stellar id: astronomy owns this, not solar
    q("compare the sun to KIC 8462852", "astronomy", AST,
      "astronomy.isolation_forest", ADVERSARIAL),
    q("are there flares on KIC 5358624", "astronomy", AST,
      "astronomy.isolation_forest", ADVERSARIAL),
    q("does KIC 8544996 show solar-like activity", "astronomy", AST,
      "astronomy.isolation_forest", ADVERSARIAL),
    # finance words that look astronomical
    q("RELIANCE volatility spiked, any anomalies", "finance", FIN,
      "finance.anomaly_detectors", ADVERSARIAL),
    q("was there a flare up in TCS volatility", "finance", FIN,
      "finance.anomaly_detectors", ADVERSARIAL),
    # count vs probability, stated ambiguously
    q("how many flares are likely tomorrow", "solar", SOL_H,
      "solar.hawkes_expected_count", ADVERSARIAL),
    q("what are the odds of several flares", "solar", SOL,
      "solar.hawkes_probability", ADVERSARIAL),
    # forecast vs evaluate, minimal difference in wording
    q("how good will RELIANCE do next month", "finance", FIN_H,
      "finance.gpr_logged_forecast", ADVERSARIAL),
    q("how good is the model at RELIANCE", "finance", FIN,
      "finance.arima_backtest", ADVERSARIAL),
    # lookup vs detect, minimal difference
    q("what's unusual about KIC 5358624's planets", "astronomy", AST,
      "astronomy.koi_lookup", ADVERSARIAL),
]

QUERIES.extend(ADVERSARIAL_CASES)


def by_category(category: str) -> list:
    return [x for x in QUERIES if x["category"] == category]


def stats() -> dict:
    from collections import Counter
    return {
        "n_queries": len(QUERIES),
        "by_category": dict(Counter(x["category"] for x in QUERIES)),
        "by_domain": dict(Counter(x["domain"] for x in QUERIES)),
        "by_expected": dict(Counter(str(x["expected"]) for x in QUERIES)),
        "n_no_model_expected": sum(1 for x in QUERIES if x["expected"] is None),
    }


if __name__ == "__main__":
    import json

    s = stats()
    print(json.dumps({k: v for k, v in s.items() if k != "by_expected"}, indent=2))
    print("\nper expected model:")
    for model_id, n in sorted(s["by_expected"].items(), key=lambda kv: -kv[1]):
        print(f"  {model_id:<34}{n}")
