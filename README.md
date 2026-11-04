# Dry Bean Classification (ML Assignment 2)

> Work in progress. The deployment link gets added once the app is live.

## 1. Problem statement
Dry beans look a lot alike to the human eye, yet a batch that is mixed up with the wrong variety is a real headache for sellers and seed producers. The goal here is to see whether the shape measurements of a single bean, all taken from a photo, are enough to tell its variety apart. This is a 7-class classification problem. I train five classic models on the same data, compare them on the same held-out test set, and put them behind a small Streamlit app so anyone can try them on their own CSV.

## 2. Dataset description
- **Source:** [UCI Machine Learning Repository – Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) (Koklu & Ozkan, 2020).
- **Size:** 13,611 beans, 16 numeric features, 1 target column (`Class`). This is above the assignment minimums of 12 features and 500 rows.
- **Features:** area, perimeter, major and minor axis length, aspect ratio, eccentricity, convex area, equivalent diameter, extent, solidity, roundness, compactness and four shape factors. All of them were computed from high-resolution images of the beans.
- **Target:** seven varieties – Barbunya, Bombay, Cali, Dermason, Horoz, Seker and Sira.
- **Class balance:** Dermason is the biggest group (3,546 beans) and Bombay the smallest (522), so the classes are noticeably unbalanced. Because of that, precision, recall and F1 are **macro-averaged**, which gives Bombay the same weight as Dermason. AUC is one-vs-rest, also macro-averaged.
- **Split:** 80/20, stratified, with a fixed seed. 10,888 rows go to `data/train_data.csv` and 2,723 rows to `test_data.csv`. No values are missing, so no imputation was needed.
- **Preprocessing:** Logistic Regression, KNN and SVM are saved as pipelines with a `StandardScaler` in front, so the saved files accept raw feature values. Decision Tree and Naive Bayes don't need scaling.

## 3. GitHub repository
https://github.com/AnandAbhigyan/problemSolving

## 4. Models used

All numbers are measured on `test_data.csv` (2,723 beans the models never saw during training). You can reproduce them with `python train_models.py`.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9243 | 0.9947 | 0.9369 | 0.9366 | 0.9366 | 0.9086 |
| Decision Tree | 0.9097 | 0.9803 | 0.9192 | 0.9174 | 0.9179 | 0.8909 |
| KNN | 0.9221 | 0.9884 | 0.9372 | 0.9351 | 0.9361 | 0.9059 |
| Naive Bayes | 0.7631 | 0.9668 | 0.7683 | 0.7645 | 0.7636 | 0.7141 |
| Support Vector Machine | 0.9321 | 0.9909 | 0.9453 | 0.9440 | 0.9446 | 0.9179 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Surprisingly strong for such a simple model: 92.4% accuracy and the best AUC of the five (0.9947). The shape features separate the beans nearly linearly, so there isn't much a fancier model can add. It struggles a little with Sira (F1 0.87), its weakest class. |
| Decision Tree | The weakest of the "good" models (91.0% accuracy). It keeps up on easy classes like Bombay but drops on Barbunya (F1 0.87) and Cali (0.90), which have overlapping measurements. Capping the depth at 8 stops it from memorising the training set, but a single tree is still a bit jumpy. Its big plus is that it's easy to explain. |
| KNN | Basically tied with Logistic Regression (92.2% accuracy, F1 0.936) once the features are scaled. Without scaling, the large-valued Area and ConvexArea would dominate the distance, which is why it sits behind a `StandardScaler`. Predictions are slower, and the saved model is the largest of the five `.joblib` files because it has to keep the whole training set. |
| Naive Bayes | Clearly the odd one out at 76.3% accuracy. The model assumes the features are independent, but these 16 are heavily correlated (area, perimeter, axis lengths and convex area all describe the same blob), so that assumption hurts, particularly for Barbunya (F1 0.53). It's still decent at ranking classes (AUC 0.967), meaning the probabilities are sensible, but its hard decisions are often wrong. |
| Support Vector Machine | Best on every metric except AUC: 93.2% accuracy, F1 0.945 and MCC 0.918. The RBF kernel can draw the curved boundaries that the linear model can't, which helps most on Barbunya and Cali. The price is training time, and the probabilities needed for AUC come from an extra calibration step. |

**Overall winner:** the **Support Vector Machine**. It leads on accuracy, precision, recall, F1 and MCC. Logistic Regression has a slightly higher AUC, so if you only cared about probability ranking or needed a tiny, fast model, it would be a very fair pick. The gap between the top three is only about one percentage point, so I wouldn't read too much into small differences; the clear lesson is that Naive Bayes is a poor fit for this correlated data. Sira is the toughest class for the four better models (F1 of 0.86–0.87), while Bombay is separated almost perfectly by all five (F1 of 0.995–1.0).

## 5. Streamlit app
_Live link to be added after deployment on Streamlit Community Cloud._

The app lets you:
1. upload a CSV (use `test_data.csv`; it needs the 16 feature columns plus `Class`),
2. choose one of the five saved models from a dropdown,
3. see accuracy, AUC, precision, recall, F1 and MCC,
4. look at the confusion matrix and classification report,
5. compare all five models on the uploaded file at once.

The models are loaded from `model/`; nothing is retrained in the app.

Run it locally with:

```
pip install -r requirements.txt
streamlit run app.py
```

## Project layout

```
problemSolving/
|-- app.py              Streamlit front end
|-- evaluation.py       the six metrics, shared by training and the app
|-- train_models.py     trains and saves the five models, writes the metrics table
|-- fetch_data.py       converts the raw UCI ARFF file to CSV
|-- split_data.py       80/20 stratified train/test split
|-- requirements.txt
|-- README.md
|-- test_data.csv       held-out data for the app
|-- data/               full and training data
`-- model/              saved classifiers, feature list and metrics summary
```
