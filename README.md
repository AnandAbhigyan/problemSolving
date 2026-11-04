# Dry Bean Classification (ML Assignment 2)

> Work in progress. The observations and the deployment link get filled in as the project moves along.

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
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |

**Overall winner:** _to be decided once the numbers are in._

## 5. Streamlit app
_Live link to be added after deployment._

## Planned project layout

```
problemSolving/
|-- app.py              Streamlit front end
|-- requirements.txt
|-- README.md
|-- test_data.csv       held-out data for the app
|-- data/               full and training data
`-- model/              saved classifiers
```
