"""Trains the classifiers on data/train_data.csv and saves them under model/."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import TARGET, compute_metrics

SEED = 7
MODEL_DIR = Path("model")

# Each entry is a full pipeline, so the saved file can take raw feature values.
MODELS = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)
    ),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "Naive Bayes": GaussianNB(),
    # SVC has no native probabilities; calibration gives us them for the AUC score
    "Support Vector Machine": make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            SVC(kernel="rbf", C=10, random_state=SEED), ensemble=False, cv=5
        ),
    ),
}


def file_name(model_name):
    return model_name.lower().replace(" ", "_") + ".joblib"


def main():
    train = pd.read_csv("data/train_data.csv")
    test = pd.read_csv("test_data.csv")
    X_train, y_train = train.drop(columns=TARGET), train[TARGET]
    X_test, y_test = test.drop(columns=TARGET), test[TARGET]

    MODEL_DIR.mkdir(exist_ok=True)
    results = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / file_name(name))
        results[name], _ = compute_metrics(model, X_test, y_test)
        print(f"{name:22s}", results[name])

    table = pd.DataFrame(results).T
    table.index.name = "ML Model Name"
    table.to_csv(MODEL_DIR / "metrics_summary.csv")
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(list(X_train.columns)))


if __name__ == "__main__":
    main()
