"""The multi-agent system: a LangGraph StateGraph wiring together the
Orchestrator/Router, Finance, Sentiment, Finance-Anomaly, Astronomy, and
Explainer agents described in TECHNICAL_PLAN.md Section 1.

The finance branch is CONDITIONAL on the task, not a fixed chain. It used to
be three unconditional edges:

    finance -> sentiment -> finance_anomaly -> explain

which meant every finance question ran everything. Asking "what is Reliance's
RSI?" -- a number already sitting in the dataframe finance_node builds --
downloaded two years of prices, fitted ARIMA, trained XGBoost, fetched RSS
and Reddit headlines, scored each one through FinBERT, trained XGBoost a
SECOND time with sentiment, and ran five anomaly detectors. The answer was
one number the first node already had.

Now agents/model_selector.py classifies the task and the graph skips what
that task does not need. The edges below are the routing; which task needs
which node is decided in _needs_sentiment / _needs_anomaly, kept as named
functions so the policy is readable rather than buried in a lambda.

    route --(finance)--> finance --> [sentiment?] --> [finance_anomaly?] --> explain --> END
          |-(astronomy)-> astronomy ------------------------------------------> explain --> END
          `-(solar)----> solar_flare -----------------------------------------> explain --> END

Solar is in the graph now too. It was previously reachable only through the
chatbot, which meant two orchestration paths that could disagree with each
other.
"""

from langgraph.graph import END, StateGraph

from agents.model_registry import (COMPARE_MODELS, DETECT_ANOMALY, SCORE_SENTIMENT)
from agents.nodes.astronomy_node import astronomy_node
from agents.nodes.explainer_node import explainer_node
from agents.nodes.finance_anomaly_node import finance_anomaly_node
from agents.nodes.finance_node import finance_node
from agents.nodes.router import route_node
from agents.nodes.sentiment_node import sentiment_node
from agents.nodes.solar_flare_node import solar_flare_node
from agents.state import AgentState

# Tasks whose answer genuinely depends on news sentiment. Everything else
# skips FinBERT, which is the single slowest step in the finance branch.
_SENTIMENT_TASKS = {SCORE_SENTIMENT, COMPARE_MODELS, DETECT_ANOMALY}

# Only anomaly questions need the five detectors.
_ANOMALY_TASKS = {DETECT_ANOMALY}


def _needs_sentiment(state: AgentState) -> str:
    """An UNKNOWN task runs the full pipeline.

    Deliberate: when the classifier has not recognised the question, the old
    exhaustive behaviour is the safe answer. Skipping work on a guess risks
    answering without data the user actually asked about, which is a worse
    failure than being slow.
    """
    task = state.get("task")
    return "sentiment" if (task is None or task in _SENTIMENT_TASKS) else "explain"


def _needs_anomaly(state: AgentState) -> str:
    task = state.get("task")
    return "finance_anomaly" if (task is None or task in _ANOMALY_TASKS) else "explain"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("route", route_node)
    graph.add_node("finance", finance_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("finance_anomaly", finance_anomaly_node)
    graph.add_node("astronomy", astronomy_node)
    graph.add_node("solar", solar_flare_node)
    graph.add_node("explain", explainer_node)

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        lambda state: state["domain"],
        {"finance": "finance", "astronomy": "astronomy", "solar": "solar"},
    )

    graph.add_conditional_edges("finance", _needs_sentiment,
                                {"sentiment": "sentiment", "explain": "explain"})
    graph.add_conditional_edges("sentiment", _needs_anomaly,
                                {"finance_anomaly": "finance_anomaly", "explain": "explain"})
    graph.add_edge("finance_anomaly", "explain")
    graph.add_edge("astronomy", "explain")
    graph.add_edge("solar", "explain")
    graph.add_edge("explain", END)

    return graph.compile()


def run(ticker: str | None = None, target: str | None = None,
        question_focus: str | None = None, task: str | None = None,
        domain: str | None = None) -> dict:
    """Invoke the graph.

    `task` is what makes the finance branch conditional; omitting it keeps
    the original exhaustive behaviour, so every existing caller is unaffected.
    """
    app = build_graph()
    initial_state = {"trace": []}
    if ticker:
        initial_state["ticker"] = ticker
    if target:
        initial_state["target"] = target
    if question_focus:
        initial_state["question_focus"] = question_focus
    if task:
        initial_state["task"] = task
    if domain:
        initial_state["domain"] = domain
    return app.invoke(initial_state)


if __name__ == "__main__":
    import sys
    import time

    query = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    task = sys.argv[2] if len(sys.argv) > 2 else None

    if query.upper().startswith(("KIC", "TIC", "EPIC")):
        started = time.time()
        result = run(target=query, task=task)
    else:
        started = time.time()
        result = run(ticker=query, task=task)

    print(f"\n=== TRACE (task={task or 'unset -- full pipeline'}, "
          f"{time.time() - started:.1f}s) ===")
    for line in result["trace"]:
        print(line)
    print("\n=== REPORT ===")
    print(result["report"])
