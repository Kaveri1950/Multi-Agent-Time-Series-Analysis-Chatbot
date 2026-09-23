"""News/social sentiment, entirely free:

- Headlines: Yahoo Finance RSS + Google News RSS (no API key, no rate limit
  headaches). Reddit via PRAW is supported too, but needs a free Reddit
  developer app (client_id/secret) -- optional, degrades gracefully if
  REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET env vars aren't set.
- Scoring: FinBERT (ProsusAI/finbert), a HuggingFace model fine-tuned for
  financial text, run locally via `transformers` -- no OpenAI/Cohere
  embedding or sentiment API involved.
"""

import os
from datetime import datetime, timezone
from urllib.parse import quote

import feedparser
import pandas as pd

YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en"

_finbert_pipeline = None


def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert",
        )
    return _finbert_pipeline


def fetch_headlines_rss(ticker: str, company_query: str | None = None, max_items: int = 25) -> pd.DataFrame:
    # quote() the query params -- an unescaped space (e.g. company_query="Reliance
    # Industries") is an invalid raw URL character and crashes urllib with
    # "URL can't contain control characters" before the request is even sent.
    rows = []
    for url in (YAHOO_RSS.format(ticker=quote(ticker)), GOOGLE_NEWS_RSS.format(query=quote(company_query or ticker))):
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_items]:
            published = entry.get("published_parsed")
            ts = datetime(*published[:6], tzinfo=timezone.utc) if published else None
            rows.append({"ticker": ticker, "headline": entry.get("title", ""), "published": ts, "source": feed.feed.get("title", url)})
    return pd.DataFrame(rows).drop_duplicates(subset="headline")


def fetch_reddit_posts(query: str, subreddits: str = "stocks+wallstreetbets+investing", limit: int = 25) -> pd.DataFrame:
    """Optional. Requires a free Reddit 'script' app -- create one at
    https://www.reddit.com/prefs/apps and set REDDIT_CLIENT_ID /
    REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT env vars. Returns an empty
    frame (not an error) if credentials aren't configured, so the rest of
    the pipeline keeps working with RSS-only sentiment."""
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        return pd.DataFrame(columns=["ticker", "headline", "published", "source"])

    import praw
    reddit = praw.Reddit(
        client_id=client_id, client_secret=client_secret,
        user_agent=os.environ.get("REDDIT_USER_AGENT", "btp-multiagent-timeseries/0.1"),
    )
    rows = []
    for submission in reddit.subreddit(subreddits).search(query, limit=limit, sort="new"):
        rows.append({
            "ticker": query, "headline": submission.title,
            "published": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
            "source": f"reddit/r/{submission.subreddit.display_name}",
        })
    return pd.DataFrame(rows)


def score_sentiment(headlines_df: pd.DataFrame) -> pd.DataFrame:
    """FinBERT gives 'positive'/'negative'/'neutral' + confidence. We convert
    to a single signed score in [-1, 1]: +confidence for positive, -confidence
    for negative, 0 for neutral."""
    if headlines_df.empty:
        return headlines_df.assign(sentiment_label=[], sentiment_score=[])

    clf = _get_finbert()
    texts = headlines_df["headline"].tolist()
    results = clf(texts, truncation=True)

    sign = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    scores = [sign[r["label"].lower()] * r["score"] for r in results]
    labels = [r["label"].lower() for r in results]

    out = headlines_df.copy()
    out["sentiment_label"] = labels
    out["sentiment_score"] = scores
    return out


def daily_sentiment_series(scored_df: pd.DataFrame) -> pd.Series:
    """Aggregate to one sentiment score per calendar day, for joining onto
    the price/technical-indicator frame in finance/price_model.py."""
    if scored_df.empty or "published" not in scored_df or scored_df["published"].isna().all():
        return pd.Series(dtype=float)
    df = scored_df.dropna(subset=["published"]).copy()
    df["date"] = pd.to_datetime(df["published"]).dt.tz_localize(None).dt.normalize()
    return df.groupby("date")["sentiment_score"].mean()


def headlines_near_date(headlines_df: pd.DataFrame, target_date, window_days: int = 3) -> pd.DataFrame:
    """Filters already-fetched headlines to ones published within
    `window_days` of `target_date` -- for explaining a specific flagged
    anomaly day (finance/anomaly_detector.py) with real news rather than
    just the statistical reasons it was flagged.

    Honest limitation, by design: Yahoo/Google News RSS only serve
    CURRENT headlines, not a historical archive. This will find real
    matches for a recent anomaly, and correctly find nothing for an older
    one -- callers should show that as "no news available near this date",
    not silently fall back to unrelated recent headlines or invent a cause."""
    if headlines_df.empty or "published" not in headlines_df.columns:
        return headlines_df.iloc[0:0]

    target = pd.Timestamp(target_date)
    if target.tzinfo is not None:
        target = target.tz_localize(None)

    published = pd.to_datetime(headlines_df["published"])
    if published.dt.tz is not None:
        published = published.dt.tz_localize(None)

    within_window = (published - target).abs() <= pd.Timedelta(days=window_days)
    return headlines_df[within_window.values]


def get_ticker_sentiment(ticker: str, company_query: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """One-call convenience wrapper: RSS + (optional) Reddit -> FinBERT -> daily series."""
    headlines = fetch_headlines_rss(ticker, company_query=company_query)
    reddit_posts = fetch_reddit_posts(company_query or ticker)
    combined = pd.concat([headlines, reddit_posts], ignore_index=True) if not reddit_posts.empty else headlines
    scored = score_sentiment(combined)
    return scored, daily_sentiment_series(scored)


if __name__ == "__main__":
    scored, daily = get_ticker_sentiment("AAPL", company_query="Apple")
    print(f"Fetched {len(scored)} headlines/posts")
    if not scored.empty:
        print(scored[["headline", "sentiment_label", "sentiment_score"]].head(10).to_string(index=False))
    print("\nDaily aggregated sentiment:")
    print(daily.tail(10))
