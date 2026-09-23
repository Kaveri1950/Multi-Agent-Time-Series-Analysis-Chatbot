import streamlit as st

from agents.chat_agent import answer_query
from dashboard import chat_store

st.title("💬 Multi-Agent Chatbot")
st.caption("Ask about an Indian stock/index/mutual fund's price/RSI/sentiment/forecast, ask about a Kepler "
           "light curve's anomalies, or ask a general finance/astronomy question. Multi-turn: follow-ups "
           "reuse the last ticker/target/fund. If a company name matches multiple real tickers (e.g. "
           "\"reliance\"), you'll be asked to pick one. You can also attach a light-curve plot image "
           "(JPG/PNG) using the 📎 button and ask about it directly.")

_CONVERSATION_KEYS = ("last_domain", "last_ticker", "last_target", "last_mutual_fund",
                      "_pipeline_cache", "_forecast_cache", "_mf_cache",
                      "_pending_disambiguation", "pinned_domain")


def _reset_conversation_state():
    """Clear everything that belongs to ONE conversation.

    The pipeline cache is included deliberately: it is keyed by
    domain+ticker, so carrying it across a conversation switch would let a
    new chat silently answer from data fetched in an old one -- fast, and
    wrong if the user expected a fresh look.
    """
    st.session_state.messages = []
    for k in _CONVERSATION_KEYS:
        st.session_state.pop(k, None)


with st.sidebar:
    st.subheader("Settings")
    backend = st.selectbox("LLM backend", ["ollama", "openai", "claude"], key="chat_backend",
                            help="ollama = free/local (default). openai/claude = paid, final-demo only.")
    model = st.text_input("Model", "llama3.2:3b" if backend == "ollama" else "", key="chat_model")

    st.divider()
    st.subheader("Chats")

    _store_ok, _store_reason = chat_store.is_available()
    if not _store_ok:
        # History is a convenience, not a dependency -- say so plainly and
        # carry on rather than breaking the chatbot over a dead database.
        st.caption(f"⚠️ History unavailable ({_store_reason}). This session still works, "
                   f"but conversations will not be saved.")
        if st.button("Clear conversation", width="stretch"):
            _reset_conversation_state()
            st.rerun()
    else:
        if st.button("➕ New chat", width="stretch"):
            st.session_state.conversation_id = None
            _reset_conversation_state()
            st.rerun()

        conversations = chat_store.list_conversations()
        current = st.session_state.get("conversation_id")

        for conv in conversations:
            is_current = conv["id"] == current
            col_open, col_del = st.columns([6, 1])
            label = ("▸ " if is_current else "") + conv["title"]
            if col_open.button(label, key=f"open_{conv['id']}", width="stretch",
                               type="primary" if is_current else "secondary",
                               help=f"{conv['n_messages']} messages"):
                st.session_state.conversation_id = conv["id"]
                # Load the transcript, but reset the per-conversation caches:
                # the stored history is text only, so nothing carries over
                # that a fresh answer would need.
                _reset_conversation_state()
                st.session_state.messages = chat_store.load_messages(conv["id"])
                st.rerun()
            if col_del.button("🗑", key=f"del_{conv['id']}", help="Delete this chat"):
                chat_store.delete_conversation(conv["id"])
                if is_current:
                    st.session_state.conversation_id = None
                    _reset_conversation_state()
                st.rerun()

        if not conversations:
            st.caption("No saved chats yet — ask something to start one.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None


def _charts_of(payload):
    """Every figure attached to a turn.

    An astronomy answer now carries three (raw curve, flagged regions, and a
    zoom on the strongest flag) because one chart cannot show both a
    four-year baseline and a twelve-hour event. "charts" is the list;
    "chart" is kept as the single-figure fallback so finance and solar turns,
    and any history saved before this change, still render.
    """
    charts = payload.get("charts")
    if charts:
        return [c for c in charts if c is not None]
    single = payload.get("chart")
    return [single] if single is not None else []


def _render_trace(trace, retrieved_context):
    if trace:
        with st.expander("Agent trace / retrieved context"):
            for line in trace:
                st.code(line, language=None)
            for item in retrieved_context or []:
                # RAG hits arrive as {"score": .., "text": ..}; the astronomy
                # and solar paths attach plain strings (the grounded fact
                # sheet the model wrote from). Assuming a dict crashed the
                # whole page with "'str' object has no attribute 'get'".
                if isinstance(item, dict):
                    st.write(f"score={item.get('score', 0):.3f} — {item.get('text', item)}")
                else:
                    st.code(str(item), language=None)


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("image_name"):
            st.caption(f"📎 {msg['image_name']}")
        if msg["content"]:
            st.write(msg["content"])
        for j, fig in enumerate(_charts_of(msg)):
            st.plotly_chart(fig, use_container_width=True, key=f"history_chart_{i}_{j}")
        _render_trace(msg.get("trace"), msg.get("retrieved_context"))

chat_value = st.chat_input(
    "e.g. \"what's the price for reliance in the next 30 days\", \"how is nifty 50 doing\", "
    "\"tell me about SBI Large Cap fund returns\", \"check KIC 11446443 for anomalies\", "
    "or attach a light-curve image and ask \"any anomalies here?\"",
    accept_file=True,
    file_type=["png", "jpg", "jpeg"],
)

if chat_value:
    text = chat_value.text
    uploaded = chat_value.files[0] if chat_value.files else None
    image_bytes = uploaded.getvalue() if uploaded else None
    image_name = uploaded.name if uploaded else None

    st.session_state.messages.append({"role": "user", "content": text, "image_name": image_name})
    # A conversation row is created lazily, on the first real message, so
    # clicking "New chat" and changing your mind does not leave an empty
    # conversation cluttering the sidebar.
    if st.session_state.get("conversation_id") is None:
        st.session_state.conversation_id = chat_store.create_conversation(text)
    chat_store.add_message(st.session_state.conversation_id, "user", text,
                           image_name=image_name)

    with st.chat_message("user"):
        if image_name:
            st.caption(f"📎 {image_name}")
            st.image(image_bytes, width=300)
        if text:
            st.write(text)

    with st.chat_message("assistant"):
        spinner_msg = ("Extracting the light curve from the image and running anomaly detection..." if image_bytes
                       else "Thinking (may take longer the first time a ticker/target is analyzed, or for a price forecast)...")
        with st.spinner(spinner_msg):
            try:
                result = answer_query(text, st.session_state, backend=backend, model=model or None,
                                       image_bytes=image_bytes, image_name=image_name)
            except Exception as exc:  # noqa: BLE001
                result = {"answer": f"Something went wrong: {exc}", "trace": [], "retrieved_context": [], "chart": None}

        st.write(result["answer"])
        for j, fig in enumerate(_charts_of(result)):
            st.plotly_chart(fig, use_container_width=True,
                            key=f"live_chart_{len(st.session_state.messages)}_{j}")
        _render_trace(result.get("trace"), result.get("retrieved_context"))

    st.session_state.messages.append({
        "role": "assistant", "content": result["answer"], "chart": result.get("chart"),
        "charts": result.get("charts"),
        "trace": result.get("trace", []), "retrieved_context": result.get("retrieved_context", []),
    })
    chat_store.add_message(st.session_state.conversation_id, "assistant", result["answer"],
                           trace=result.get("trace"),
                           retrieved_context=result.get("retrieved_context"),
                           domain=result.get("domain"))
