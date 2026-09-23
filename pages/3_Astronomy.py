import streamlit as st

from agents.astronomy_dashboard_agent import DEFAULT_CONFIG, KNOWN_SECTIONS, parse_astronomy_preference
from dashboard.charts import dataset_anomaly_overview_chart
from dashboard.sections import render_astronomy_pro

st.title("🔭 Astronomy — Light Curve Anomaly Detection")
st.caption("Real Kepler/TESS data via lightkurve (MAST archive), or upload your own CSV with a 'flux' column. "
           "Five-step flow: light curve → description → anomaly detection → normalized view → why it was flagged.")

if "astro_dashboard_config" not in st.session_state:
    st.session_state.astro_dashboard_config = dict(DEFAULT_CONFIG)
config = st.session_state.astro_dashboard_config

with st.sidebar:
    st.header("Adaptive layout")
    pref_text = st.text_input("Describe how you want this page to look",
                               placeholder='e.g. "just show me the anomaly explanations for top 3"',
                               key="astro_dash_pref")
    col_a, col_b = st.columns(2)
    apply_clicked = col_a.button("Apply", type="primary", key="astro_apply")
    reset_clicked = col_b.button("Reset", key="astro_reset")

    if apply_clicked and pref_text.strip():
        with st.spinner("Interpreting your preference..."):
            st.session_state.astro_dashboard_config = parse_astronomy_preference(pref_text, current_config=config)
        st.rerun()
    if reset_clicked:
        st.session_state.astro_dashboard_config = dict(DEFAULT_CONFIG)
        st.rerun()

    st.caption(f"💡 {config.get('note', '')}")

    st.divider()
    st.subheader("Manual override")
    manual_sections = st.multiselect("Visible steps", KNOWN_SECTIONS, default=config["visible_sections"],
                                      key="astro_manual_sections")
    manual_size = st.selectbox("Layout size", ["compact", "normal", "large"],
                                index=["compact", "normal", "large"].index(config["layout_size"]),
                                key="astro_manual_size")
    manual_top_n = st.slider("Anomalies to explain in detail", 1, 10, config.get("top_n_anomalies", 5),
                              key="astro_manual_top_n")
    if (manual_sections != config["visible_sections"] or manual_size != config["layout_size"]
            or manual_top_n != config.get("top_n_anomalies", 5)):
        config["visible_sections"] = manual_sections
        config["layout_size"] = manual_size
        config["top_n_anomalies"] = manual_top_n

with st.expander("📦 Downloaded dataset overview (ask the Chatbot about this too)"):
    try:
        from astronomy.dataset import load_index

        index_df = load_index()
        st.plotly_chart(dataset_anomaly_overview_chart(index_df.sort_values("anomaly_rate", ascending=False)),
                         use_container_width=True, key="dataset_overview_chart")
        st.dataframe(index_df[["target", "source", "n_points", "n_flagged_anomalies", "anomaly_rate"]],
                     use_container_width=True, hide_index=True)
        real_count = index_df["source"].str.startswith("real").sum()
        st.caption(f"{real_count} of {len(index_df)} targets used real MAST data; the rest fell back to a "
                   "labeled synthetic curve (network/archive issue for that specific target).")
    except FileNotFoundError:
        st.info("No dataset built yet. Run `python -m astronomy.dataset` to download and index a batch of "
                "Kepler targets, then this section (and the Chatbot's dataset-level queries) will populate.")

default_target = config.get("focus_target") or "KIC 11446443"
col1, col2 = st.columns([2, 1])
with col1:
    target = st.text_input("Kepler/TESS Target ID", default_target, key="astro_target",
                            help="Try: KIC 11446443 (confirmed hot Jupiter host), KIC 8462852 (Tabby's Star)")
with col2:
    uploaded = st.file_uploader("...or upload a light curve CSV, or a photo/screenshot (JPG/PNG)",
                                 type=["csv", "png", "jpg", "jpeg"], key="astro_upload",
                                 help="An image only recovers the plotted curve's shape, not physical time/flux "
                                      "units -- best for a clearly cropped plot with a visually distinct curve.")

if target or uploaded:
    try:
        render_astronomy_pro(target or (uploaded.name if uploaded else "uploaded"), uploaded_file=uploaded,
                              key_prefix="astro", config=config)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't analyze '{target}': {exc}")
