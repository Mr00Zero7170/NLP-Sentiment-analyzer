import pandas as pd
import streamlit as st

from src.sentiment_analyzer import BertSentimentAnalyzer

REPO_URL = "https://github.com/Mr00Zero7170/NLP-Sentiment-analyzer"

MODEL_REGISTRY = {
    "DistilBERT SST-2 (Fast)": {
        "id": "distilbert-base-uncased-finetuned-sst-2-english",
        "dataset": "Stanford Sentiment Treebank (SST-2)",
        "accuracy": "~91-93% (reported on SST-2)",
        "f1": "~91-93% (binary sentiment)",
        "architecture": "DistilBERT encoder + classification head",
    },
    "RoBERTa Large Sentiment (High Accuracy)": {
        "id": "siebert/sentiment-roberta-large-english",
        "dataset": "Large English sentiment corpora (fine-tuned)",
        "accuracy": "High (model-card reported)",
        "f1": "High (model-card reported)",
        "architecture": "RoBERTa-large + classification head",
    },
}

EXAMPLES = [
    "I love how smooth and reliable this product feels.",
    "The experience was frustrating and full of bugs.",
    "It is okay overall, but support could be better.",
]


@st.cache_resource
def get_analyzer(model_name: str) -> BertSentimentAnalyzer:
    return BertSentimentAnalyzer(model_name=model_name)


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(
                1200px 600px at 10% -20%,
                #182436 0%,
                #0b1019 45%,
                #070b12 100%
            );
            color: #e8eefc;
        }
        .panel {
            border: 1px solid #26344a;
            border-radius: 14px;
            padding: 18px;
            background: rgba(16, 24, 38, 0.65);
            backdrop-filter: blur(2px);
        }
        .result-box {
            border-radius: 12px;
            padding: 14px;
            border: 1px solid #2f3f58;
            margin-bottom: 12px;
            background: rgba(14, 20, 31, 0.9);
        }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .badge-pos {
            background: rgba(46, 204, 113, 0.18);
            color: #7dffbb;
            border: 1px solid rgba(46, 204, 113, 0.45);
        }
        .badge-neg {
            background: rgba(255, 99, 99, 0.18);
            color: #ff9a9a;
            border: 1px solid rgba(255, 99, 99, 0.45);
        }
        .metric-label { color: #96a5bf; font-size: 0.86rem; }
        .metric-value { font-size: 1.2rem; font-weight: 650; }
        .github-cta {
            margin-top: 8px;
            padding: 8px 0 2px 0;
            border-top: 1px solid #253248;
        }
        @media (max-width: 900px) {
            .panel { padding: 14px; }
            .result-box { padding: 12px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_state() -> None:
    st.session_state.setdefault("input_text", "")
    st.session_state.setdefault("run_inference", False)
    st.session_state.setdefault("pending_text", "")
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("error", "")


def render_left_panel() -> tuple[str, str]:
    model_label = st.selectbox("Model", list(MODEL_REGISTRY.keys()))
    model_name = MODEL_REGISTRY[model_label]["id"]

    st.markdown("**Try an example**")
    ex_cols = st.columns(3)
    for idx, text in enumerate(EXAMPLES):
        if ex_cols[idx].button(f"Example {idx + 1}", use_container_width=True):
            st.session_state.input_text = text

    st.text_area(
        "Input text",
        key="input_text",
        height=190,
        placeholder="Enter text to run sentiment inference...",
    )

    run_disabled = st.session_state.run_inference
    if st.button("Run Inference", type="primary", disabled=run_disabled, use_container_width=True):
        input_text = st.session_state.input_text.strip()
        if not input_text:
            st.session_state.error = "Please enter text before running inference."
        else:
            st.session_state.error = ""
            st.session_state.pending_text = input_text
            st.session_state.run_inference = True
            st.rerun()

    if st.session_state.error:
        st.warning(st.session_state.error)

    with st.expander("Model Information"):
        info = MODEL_REGISTRY[model_label]
        st.markdown(f"**Dataset:** {info['dataset']}")
        st.markdown(f"**Accuracy:** {info['accuracy']}")
        st.markdown(f"**F1-score:** {info['f1']}")
        st.markdown(f"**Architecture:** {info['architecture']}")

    return model_name, model_label


def render_result(result: dict) -> None:
    label = str(result["label"]).upper()
    confidence = float(result["score"]) * 100
    is_positive = "POS" in label
    badge_class = "badge-pos" if is_positive else "badge-neg"

    st.markdown(
        f"""
        <div class="result-box">
            <div class="metric-label">Sentiment</div>
            <div style="margin: 6px 0 10px 0;">
                <span class="badge {badge_class}">{label}</span>
            </div>
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{confidence:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    probs = result.get("probabilities", {})
    if probs:
        df = pd.DataFrame(
            {"Label": list(probs.keys()), "Probability": [float(v) for v in probs.values()]}
        ).sort_values("Probability", ascending=False)
        st.caption("Probability distribution")
        st.bar_chart(df.set_index("Label"), use_container_width=True)

    st.caption("Inference payload")
    st.json(result)


def render_right_panel(model_name: str) -> None:
    if st.session_state.run_inference:
        with st.spinner("Running transformer inference..."):
            analyzer = get_analyzer(model_name)
            result = analyzer.predict_detailed(st.session_state.pending_text)
            st.session_state.last_result = result
            st.session_state.run_inference = False
        st.rerun()

    if st.session_state.last_result:
        render_result(st.session_state.last_result)
    else:
        st.info("Run inference to see sentiment label, confidence, and probability chart.")


def main() -> None:
    st.set_page_config(page_title="NLP Sentiment Analyzer", page_icon="🧠", layout="wide")
    ensure_state()
    apply_custom_css()

    st.title("NLP Sentiment Analyzer")
    st.caption("Production-style sentiment dashboard using BERT + PyTorch + Transformers")

    left_col, right_col = st.columns([1.05, 1], gap="large")

    with left_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        model_name, _ = render_left_panel()
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        render_right_panel(model_name)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="github-cta"></div>', unsafe_allow_html=True)
    st.link_button("View on GitHub", REPO_URL, use_container_width=True)


if __name__ == "__main__":
    main()
