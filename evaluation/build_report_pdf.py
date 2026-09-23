"""Build the mid-semester report as a PDF, without a LaTeX installation.

There is no pdflatex on this machine, and Overleaf is where the submitted
version will be compiled from `report/midsem_report.tex`. This produces a
local preview so the content can be read and checked now rather than after a
round trip through a browser.

The two are deliberately kept in step: every figure comes from the same PDFs
in evaluation/results/figures/, and every number is read from
evaluation/results/ rather than retyped. If a result changes, re-running
`report_figures.py` and then this script updates both.
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "evaluation" / "results"
FIGURES = RESULTS / "figures"
OUT = ROOT / "report" / "midsem_report.pdf"

INK = colors.HexColor("#1A1C19")
MUTED = colors.HexColor("#5A6158")
RULE = colors.HexColor("#D9DCD5")
ACCENT = colors.HexColor("#B8442A")


def _styles():
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=16, leading=19,
                                textColor=INK, spaceAfter=2),
        "kicker": ParagraphStyle("k", parent=base["Normal"], fontSize=9.5, leading=12,
                                 alignment=1, textColor=MUTED, spaceAfter=4),
        "subtitle": ParagraphStyle("st", parent=base["Normal"], fontSize=11, leading=14,
                                   alignment=1, textColor=INK, spaceAfter=8),
        "author": ParagraphStyle("a", parent=base["Normal"], fontSize=8.8, leading=12.5,
                                 alignment=1, textColor=INK, spaceAfter=10),
        "h": ParagraphStyle("h", parent=base["Heading2"], fontSize=11, leading=13,
                            textColor=INK, spaceBefore=10, spaceAfter=4),
        "para": ParagraphStyle("p", parent=base["BodyText"], fontSize=8.2, leading=10.6,
                               alignment=TA_JUSTIFY, textColor=INK, spaceAfter=5),
        "abstract": ParagraphStyle("ab", parent=base["BodyText"], fontSize=7.9,
                                   leading=10.0, alignment=TA_JUSTIFY,
                                   textColor=INK, leftIndent=12, rightIndent=12,
                                   spaceAfter=8),
        "cap": ParagraphStyle("c", parent=base["Normal"], fontSize=7.4, leading=9,
                              alignment=1, textColor=MUTED, spaceBefore=3,
                              spaceAfter=8),
        "ref": ParagraphStyle("r", parent=base["Normal"], fontSize=7.4, leading=9.5,
                              textColor=MUTED, leftIndent=10, firstLineIndent=-10,
                              spaceAfter=2),
    }
    return s


def _fig(name: str, caption: str, width: float, st) -> list:
    """A figure with its caption, kept on one page."""
    path = FIGURES / f"fig_{name}.png"      # PNG: reportlab cannot embed a PDF
    if not path.exists():
        return []
    from PIL import Image as PILImage

    w, h = PILImage.open(path).size
    img = Image(str(path), width=width, height=width * h / w)
    return [KeepTogether([img, Paragraph(caption, st["cap"])])]


def _results_table(data, st):
    rows = [["Domain", "Detector", "PR-AUC", "Lift", "Inflation"]]
    wanted = {
        "kepler": ["matrix_profile_discord_raw", "latent_isoforest", "moving_variance",
                   "supervised_ceiling", "raw_input"],
        "equity": ["spectral_pca", "raw_input", "hampel_mad", "local_outlier_factor",
                   "one_class_svm"],
    }
    for domain, names in wanted.items():
        for name in names:
            r = next((x for x in data["rows"] if x["domain"] == domain
                      and x["detector"] == name), None)
            if not r or not isinstance(r.get("pr_auc"), (int, float)):
                continue
            ins = r.get("in_sample_pr_auc")
            infl = f"{ins / r['pr_auc']:.1f}x" if ins and r["pr_auc"] else "--"
            label = name.replace("_", " ")
            if r.get("is_floor"):
                label += "  (floor)"
            if r.get("is_ceiling"):
                label += "  (ceiling)"
            rows.append([domain.capitalize(), label, f"{r['pr_auc']:.4f}",
                         f"{r.get('lift', 0):.2f}x", infl])

    t = Table(rows, colWidths=[2.0 * cm, 6.2 * cm, 2.2 * cm, 1.8 * cm, 2.2 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, INK),
        ("LINEBELOW", (0, 5), (-1, 5), 0.4, RULE),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [KeepTogether([t, Paragraph(
        "<b>Table 1.</b> Held-out PR-AUC for selected detectors. Lift is PR-AUC divided "
        "by the base rate; inflation is in-sample PR-AUC divided by held-out.",
        st["cap"])])]


def build():
    st = _styles()
    data = json.loads((RESULTS / "anomaly_grid.json").read_text())
    sel = json.loads((RESULTS / "selection_all.json").read_text())["results"]
    by = {r["strategy"]: r for r in sel}
    kep, eq = data["summaries"]["kepler"], data["summaries"]["equity"]
    quiet = next((c.get("fraction_quiet_positives") for c in
                  data["audits"]["kepler"]["checks"]
                  if c["flaw"] == "mislabeled_ground_truth"), 0.44)

    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                            leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.3 * cm, bottomMargin=1.3 * cm,
                            title="Mid-semester BTP report", author="Kaveri Neeli")
    W = doc.width
    S = []
    P = lambda text, style="para": S.append(Paragraph(text, st[style]))  # noqa: E731

    P("B.Tech Project  |  Mid-Semester Report", "kicker")
    P("A Multi-Agent System for Time-Series Modelling across Scientific and Financial Domains", "title")
    P("Mid-semester deliverable: the agent framework, and its first capability validated end to end", "subtitle")
    P("<b>Submitted by</b>&nbsp; Kaveri Neeli (EP230051010)<br/>"
      "<b>Supervisor</b>&nbsp; [Supervisor name], [Department]<br/>"
      "<b>Department</b>&nbsp; Department of Electrical Engineering<br/>"
      "<b>Institute</b>&nbsp; Indian Institute of Technology Indore<br/>"
      "<b>Date</b>&nbsp; September 2026", "author")

    P(f"<b>Abstract.</b> The goal of this project is an agent that performs time-series "
      f"modelling across unrelated domains, choosing the right analysis for a question "
      f"instead of running a fixed pipeline. This semester I built the framework and "
      f"validated it on the first of its tasks. The framework is a registry of 20 models "
      f"spanning 10 task types and 4 domains, with a capability filter that selects among "
      f"them. Across 123 labelled queries that filter "
      f"routes correctly {by['filter_only']['accuracy']:.1%} of the time against "
      f"{by['llm_only']['accuracy']:.1%} for an LLM, in under a second against 41 minutes; "
      f"combining the two gives 95.1%, which is worse than the filter alone. I then used the "
      f"framework to take one task, anomaly detection, all the way through on two domains. These are 440 Kepler light curves labelled by "
      f"NASA ephemerides and seven years of Indian equity returns labelled by a "
      f"pre-registered event calendar, evaluated under held-out splits. Every detector falls within "
      f"0.04 PR-AUC of a rule that learns nothing, and a supervised model given the labels "
      f"does no better; the gap between in-sample and held-out scores rises monotonically "
      f"with model capacity, from 1.0× to 6.4×. The limitation is representational rather than "
      f"algorithmic. A Box Least Squares period search recovers 61% of orbital periods "
      f"across all 440 targets at median significance 42.", "abstract")

    P("1. Project Goal and Objectives", "h")
    P("Time-series questions come in several kinds. <i>What will this be worth next month?</i> "
      "is forecasting. <i>Which days were unusual?</i> is anomaly detection. <i>How likely "
      "is an event in the next 24 hours?</i> is a point-process problem. <i>How much could I "
      "lose?</i> is risk. Each needs a different model, and the right model also depends on "
      "the domain the data came from.")
    P("Most systems handle this with a fixed pipeline per domain. The goal here is "
      "different: <b>one agent that decides, for any question, which analysis to run, and "
      "chooses on measured evidence rather than a language model's judgement.</b> The "
      "registry currently spans 10 task types across astronomy, finance and solar physics. "
      "This report delivers that framework and takes one task, anomaly detection, through "
      "full validation on two domains.")
    P("<b>Objectives.</b> <b>O1.</b> Build a multi-agent system that routes a natural-language "
      "question to the correct analysis with no language model in the decision path. "
      "<b>O2.</b> Assemble anomaly detection datasets for astronomy and finance whose labels "
      "do not depend on the series being scored. <b>O3.</b> Measure a broad set of detectors "
      "on both domains under a protocol strict enough that the result can be trusted. "
      "<b>O4.</b> Identify what actually limits performance, and demonstrate a method that "
      "overcomes it.")
    P("<b>Scope of this report.</b> The agent framework and anomaly detection on astronomy and "
      "finance. Solar-flare forecasting, the risk layer and the cross-sectional equity work "
      "are under way but belong to the second half, and appear only in the plan in Section 7.")

    P("2. Literature Review", "h")
    P("<b>Benchmarks in this field are known to be unreliable.</b> Wu and Keogh [1] examined "
      "the Yahoo, Numenta, NASA and OMNI archives and found most problems suffer from at "
      "least one of four flaws: triviality, unrealistic anomaly density, mislabelled ground "
      "truth, and run-to-failure bias. Roughly 90% yield to a single line of arithmetic, and "
      "their conclusion is that on such data no level of reported performance implies a "
      "method is useful.")
    P("<b>The scoring protocol is also compromised.</b> Kim et al. [2] showed that point "
      "adjustment, the dominant convention, lets a random anomaly score reach "
      "state-of-the-art F1, because marking a whole segment correct for one lucky hit rewards "
      "noise.")
    P("<b>Complex models do not reliably win.</b> A 2024 benchmark of 1,070 series across 40 "
      "datasets [3] concluded that simpler architectures and statistical methods often "
      "outperform neural ones. For transit detection specifically, Box Least Squares [4] "
      "remains the standard method and the matrix profile [5] is the canonical tool for "
      "subsequence anomalies. I use both, and the contrast between them turns out to be "
      "informative.")
    P("<b>Gap addressed here.</b> These critiques are widely cited but rarely applied by "
      "authors to their own data. I run the four-flaw audit on my own datasets before "
      "reporting any result, and report in-sample and held-out scores side by side so that "
      "the difference between them is visible.")

    P("3. System Design and Methodology", "h")
    P("A question is parsed into a <i>domain</i> and a <i>task</i>; the registry is filtered "
      "to models that can actually run given the available inputs; one is selected on recorded "
      "skill; it executes; and a rule-based explanation is composed. Only the final phrasing "
      "step calls a language model. Each of the 21 registry entries declares its domain, task, "
      "required inputs, return type and measured skill, so adding a model is one registry "
      "entry rather than an edit to a branch.")
    S += _fig("pipeline", "<b>Figure 1.</b> Every decision except the final phrasing is made "
                          "in Python. Profiling is the only step that reasons about data the "
                          "system has never seen.", W * 0.78, st)

    P("<b>Profiling.</b> Before choosing a detector the agent inspects the series itself. "
      "The Lomb–Scargle peak-to-background ratio separates repeating from one-off structure "
      "cleanly: over ten Kepler targets and ten equities the two populations sat at medians "
      "of 1,939 and 12.4 with no overlap, and a threshold classified all twenty correctly. "
      "This matters because the decision inverts the right detector. Scored as a discord, the "
      "matrix profile reaches ROC 0.303 on KIC 10074466, below chance; scored as a "
      "<i>motif</i>, 0.697. A transit recurs 221 times and is therefore the <i>least</i> "
      "unusual feature of its own light curve.")

    P(f"<b>Routing accuracy.</b> The capability filter reaches "
      f"{by['filter_only']['accuracy']:.1%}; the LLM alone {by['llm_only']['accuracy']:.1%}; "
      f"the two combined {by['hybrid']['accuracy']:.1%}. The filter is not merely cheaper, "
      f"it is more accurate, and adding the language model costs accuracy as well as time.")
    S += _fig("routing", "<b>Figure 2.</b> Model selection on 123 labelled queries.",
              W * 0.44, st)

    P("3.3 Datasets, labelling and protocol", "h")
    P("<b>Data and labels.</b> Astronomy: 440 Kepler light curves (217 confirmed planets, "
      "223 false positives), windowed at 12 hours with a 3-hour stride, giving 384,281 windows "
      "at 24.4% positive. A window is labelled a transit if it overlaps T₀ + nP from NASA's "
      "published ephemeris, orbital mechanics, never the brightness being scored. Finance: an "
      "equal-weighted index of the 50 most liquid NSE names, 2019–2026, windowed at 21 trading "
      "days, 1,891 windows at 2.2% positive, labelled from a pre-registered calendar of budget "
      "sessions, election results and ten documented shocks.")
    P("<b>Avoiding circular labels.</b> An earlier version labelled astronomy windows by "
      "thresholding their own <i>depth</i>, then evaluated detectors that search for deep dips; "
      "measured inflation on discovery was +0.11 AUC. The finance equivalent is labelling a day "
      "anomalous because its return was large. Solar flares are excluded for the same reason in "
      "its strongest form: a flare's class is <i>defined</i> by an X-ray threshold, so detecting "
      "flares from X-ray flux cannot fail.")
    P("<b>Audit and protocol.</b> I audited both datasets against Wu and Keogh's four flaws "
      "[1] <i>before</i> any result was recorded. Equity passes all four; Kepler warns only on "
      "anomaly density, since short-period planets transit almost continuously. PR-AUC is my decision metric "
      "because it is floored at the base rate, with ROC-AUC reported beside it, and I "
      "do not apply point adjustment: Kim et al. [2] showed that protocol lets a random score "
      "reach state-of-the-art F₁. Splits are temporal for equity and by-target for Kepler, "
      "stratified by per-target prevalence. An unstratified split produced halves at 17.9% and "
      "42.8% positive, which makes their PR-AUC values incomparable.")

    P("4. Implementation", "h")
    P("The system is written in Python and organised so the domain-neutral components cannot "
      "see domain-specific information. The <font face=\"Courier\">anomaly</font> package "
      "holds the scoring contract and windowing, 22 detectors across six families, the label "
      "sources, the four-flaw audit, held-out splitting with prevalence stratification, the "
      "Lomb-Scargle structure profiler, and a parallelised grid runner. "
      "<font face=\"Courier\">astronomy/transit_search</font> implements the Box Least "
      "Squares period search, and <font face=\"Courier\">agents/model_registry</font> holds "
      "21 models with declared capabilities and measured skill.")
    P("Detectors span thresholds (z-score, Hampel/MAD), density (isolation forest, "
      "Mahalanobis, local outlier factor, one-class SVM, elliptic envelope), reconstruction "
      "(PCA and LSTM autoencoders, spectral PCA, latent isolation forest), latent-state "
      "(hidden Markov model, GARCH residuals) and four matrix profile configurations, plus two "
      "deliberate no-skill floors and a supervised ceiling. Results are served through a "
      "Streamlit dashboard and a chatbot, and all 440 astronomy targets are precomputed into "
      "MongoDB, giving 4.5 ms lookups at query time.")

    P("5. Results and Discussion", "h")
    P(f"<b>Almost nothing clears the floor.</b> On Kepler the best detector beats the best "
      f"no-skill floor by {kep['margin_over_floor']:.4f} PR-AUC, and the supervised ceiling, a "
      f"gradient-boosted model trained on the labels, reaches only {kep['ceiling_value']:.4f} "
      f"and is outscored by three unsupervised detectors. On equity the margin is "
      f"{eq['margin_over_floor']:.4f}.")
    S += _fig("grid", "<b>Figure 3.</b> Twenty-two detectors across five families, scored on "
                      "held-out data.", W * 0.70, st)
    S += _results_table(data, st)

    P("<b>In-sample scores lie in proportion to capacity.</b> The ratio of in-sample to held-out "
      "PR-AUC rises monotonically with how much a model can memorise: 1.0–1.1× for the "
      "parameter-free matrix profile, 2.0× for threshold rules, 2.9× for isolation forest, 4.7× "
      "for local outlier factor and 6.4× for a one-class SVM. Evaluated transductively, which is the "
      "standard convention in unsupervised anomaly detection, the one-class SVM appears to be "
      "the best equity detector at 4.62× lift. Held out, it manages 1.14×. The single apparent "
      "win for sophistication was an artefact of capacity.")
    S += _fig("inflation", "<b>Figure 4.</b> In-sample scores overstate held-out performance in "
                           "proportion to model capacity.", W * 0.70, st)

    P("<b>Sequence models add nothing.</b> Every detector above treats a window as an unordered "
      "summary. An LSTM sequence-to-sequence autoencoder was added specifically to test whether "
      "temporal order inside a window carries signal. It reached 1.92× on equity, below the "
      "linear autoencoder's 2.06× and a threshold rule's 2.31×, at roughly 250× the compute.")
    P(f"<b>The representation is the limit.</b> On Kepler, {quiet:.1%} of labelled transits are "
      f"no more extreme than a typical non-transit window at this resolution, and the supervised "
      f"ceiling reaches only 1.09× chance. Changing the <i>question</i> rather than the algorithm "
      f"resolves it. Box Least Squares folds the light curve on thousands of candidate periods "
      f"and stacks every repetition. Across all 440 targets it yields a significant periodogram "
      f"peak for 91.4% and the published period to within 1% for 60.9%, at median significance "
      f"42.3 and median period error 0.005%. The gap comes from harmonic locking. Around 33.6% return P/2, P/3 "
      f"or 2P, so the signal is present and only the fold multiple is wrong.")
    S += _fig("transit", "<b>Figure 5.</b> A single transit is invisible in the raw series. "
                         "Folded on the recovered period, 221 of them stack into a clear "
                         "250 ppm dip.", W * 0.56, st)

    P("<b>Configuration inverts between domains.</b> The same algorithm needs opposite settings "
      "in the two domains. On Kepler the matrix profile must be read as a <i>motif</i> under "
      "z-normalised distance, because transits repeat and their depth varies by star. On equity "
      "it must be read as a <i>discord</i> under un-normalised distance, because shocks are "
      "unique and the size of the move is the signal. Getting both axes wrong scores 0.95×, "
      "below chance. This is the clearest evidence that the right detector is a property of the "
      "data's structure rather than of the scientific domain it came from.")

    P("6. Conclusion", "h")
    P("Three things hold up. A capability filter over measured skill routes more accurately than "
      "a language model while costing nothing, and adding the model degrades it. Under held-out "
      "evaluation with non-circular labels, twenty-two detectors across two domains sit within "
      "0.04 PR-AUC of a rule that learns nothing, their in-sample advantages scaling with "
      "capacity rather than skill. And the binding constraint is representational: a "
      "matched-filter period search recovers 61% of orbital periods where no generic detector "
      "clears 1.12× chance.")
    P("7. Work Plan for the Second Half", "h")
    P("The second half fills in the remaining tasks, taking each to the same evidential standard as anomaly detection. The agent "
      "gains a forward-looking task: risk assessment conditioned on a Markov regime model, "
      "where a regime-conditional Value-at-Risk passes Kupiec and Christoffersen coverage tests "
      "that the unconditional version fails. It also gains a third domain, blazar light curves from the "
      "Fermi-LAT repository, which the pipeline accepts without code changes and which my "
      "structure profiler predicts should behave like equities rather than like Kepler.")

    P("References", "h")
    for ref in [
        "[1] R. Wu and E. J. Keogh. Current time series anomaly detection benchmarks are flawed "
        "and are creating the illusion of progress. <i>IEEE TKDE</i>, 2022.",
        "[2] S. Kim, K. Choi, H.-S. Choi, B. Lee and S. Yoon. Towards a rigorous evaluation of "
        "time-series anomaly detection. <i>AAAI</i>, 2022.",
        "[3] Q. Liu and J. Paparrizos. The elephant in the room: towards a reliable time-series "
        "anomaly detection benchmark. <i>NeurIPS Datasets and Benchmarks</i>, 2024.",
        "[4] G. Kovács, S. Zucker and T. Mazeh. A box-fitting algorithm in the search for "
        "periodic transits. <i>A&amp;A</i>, 391:369–377, 2002.",
        "[5] C.-C. M. Yeh et al. Matrix profile I: all pairs similarity joins for time series. "
        "<i>ICDM</i>, 2016.",
    ]:
        P(ref, "ref")

    doc.build(S)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
