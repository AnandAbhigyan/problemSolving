import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

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


def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted bean type")
    ax.set_ylabel("Actual bean type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


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

left, right = st.columns(2)
labels = list(model.classes_)
with left:
    st.markdown("**Confusion matrix**")
    st.pyplot(plot_confusion(y, y_pred, labels))
with right:
    st.markdown("**Classification report**")
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

st.subheader("All five models on this file")
rows = {}
for name in MODEL_FILES:
    rows[name], _ = compute_metrics(load_model(name), X, y)
comparison = pd.DataFrame(rows).T[METRIC_COLUMNS]
comparison.index.name = "ML Model Name"
st.dataframe(
    comparison.style.format("{:.4f}").highlight_max(axis=0, color="#c8e6c9"),
    width="stretch",
)
best = comparison["F1"].idxmax()
st.success(f"Best F1 on this file: **{best}** ({comparison.loc[best, 'F1']:.4f})")