"""Shared state passed between every node in the LangGraph multi-agent graph."""

from typing import Optional, TypedDict


class AgentState(TypedDict, total=False):
    # input
    domain: str          # "finance" | "astronomy" | "solar" -- set by the user or inferred by route_node
    ticker: str          # e.g. "AAPL"        (finance path)
    target: str          # e.g. "KIC 11446443" (astronomy path)
                          # the solar path needs no key -- there is only one Sun
    question_focus: str  # optional -- set by the chatbot's intent parser (agents/chat_agent.py) so
                          # explainer_node answers the user's actual question instead of a generic summary
    task: str            # optional -- from agents/model_selector.py. Makes the finance branch
                          # conditional: unset means run everything (the original behaviour)

    # per-agent outputs
    finance_result: dict
    sentiment_result: dict
    finance_anomaly_result: dict
    astronomy_result: dict
    solar_result: dict

    # RAG + final synthesis
    retrieved_context: list
    report: str
    trace: list           # human-readable log of which agents ran, for the UI / report appendix
