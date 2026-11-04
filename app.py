import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from evaluation import METRIC_COLUMNS, TARGET, compute_metrics

MODEL_DIR = Path("model")
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


@st.cache_data
def load_feature_columns():
    return json.loads((MODEL_DIR / "feature_columns.json").read_text())


st.title("🫘 Dry Bean Type Classifier")
st.write(
    "Upload a CSV of bean measurements and see how each of the five saved models "
    "copes with it. Nothing is retrained here – the models were trained ahead of time "
    "and are only loaded to make predictions."
)

feature_cols = load_feature_columns()

with st.sidebar:
    st.header("Setup")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    model_name = st.selectbox("Pick a model", list(MODEL_FILES))
    st.caption(
        "No file handy? Use `test_data.csv` from the GitHub repo. It has the 16 "
        f"feature columns plus the `{TARGET}` column."
    )

if uploaded is None:
    st.info("👈 Upload a CSV from the sidebar to get started.")
    with st.expander("What columns does the file need?"):
        st.write(", ".join(feature_cols + [TARGET]))
    st.stop()

data = pd.read_csv(uploaded)
missing = [c for c in feature_cols + [TARGET] if c not in data.columns]
if missing:
    st.error(f"The file is missing these columns: {', '.join(missing)}")
    st.stop()

data = data.dropna(subset=feature_cols + [TARGET])
X, y = data[feature_cols], data[TARGET]

model = load_model(model_name)
unknown = set(y) - set(model.classes_)
if unknown:
    st.error(f"Found bean types the models have never seen: {sorted(unknown)}")
    st.stop()

st.subheader("Preview of your data")
st.caption(f"{len(data):,} rows, {len(feature_cols)} features")
st.dataframe(data.head(10), width="stretch")

scores, y_pred = compute_metrics(model, X, y)

st.subheader(f"Results for {model_name}")
cols = st.columns(len(METRIC_COLUMNS))
for col, metric in zip(cols, METRIC_COLUMNS):
    col.metric(metric, f"{scores[metric]:.4f}")
st.caption("Precision, recall and F1 are macro-averaged over the 7 bean types; AUC is one-vs-rest.")
