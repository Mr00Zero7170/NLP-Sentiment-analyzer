import json

import streamlit as st

from src.sentiment_analyzer import BertSentimentAnalyzer

DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


@st.cache_resource
def get_analyzer(model_name: str) -> BertSentimentAnalyzer:
    return BertSentimentAnalyzer(model_name=model_name)


def main() -> None:
    st.set_page_config(page_title="NLP Sentiment Analyzer", page_icon="💬")
    st.title("NLP Sentiment Analyzer")
    st.caption("BERT + PyTorch + Hugging Face Transformers")

    model_name = st.text_input("Model name", value=DEFAULT_MODEL)
    analyzer = get_analyzer(model_name)

    mode = st.radio("Mode", ["Single text", "Batch text"], horizontal=True)

    if mode == "Single text":
        text = st.text_area("Enter text", placeholder="Type your sentence here...")
        if st.button("Analyze sentiment", type="primary"):
            if not text.strip():
                st.warning("Please enter some text.")
                return
            result = analyzer.predict(text)
            st.success(f"Prediction: {result['label']} ({result['score']:.4f})")
            st.json(result)
        return

    batch_input = st.text_area(
        "Enter one sentence per line",
        placeholder="I love this product\nThis service is awful",
        height=180,
    )
    if st.button("Analyze batch", type="primary"):
        lines = [line.strip() for line in batch_input.splitlines() if line.strip()]
        if not lines:
            st.warning("Please enter at least one non-empty line.")
            return
        results = analyzer.predict_batch(lines)
        st.dataframe(results, use_container_width=True)
        st.download_button(
            label="Download JSON",
            data=json.dumps(results, indent=2),
            file_name="sentiment_results.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
