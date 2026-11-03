"""Splits the full Dry Bean CSV into a training file and a held-out test file.

The test file is what we upload in the Streamlit app, so the models never see it while training.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 7

df = pd.read_csv("data/dry_bean.csv")
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["Class"], random_state=SEED
)

train_df.to_csv("data/train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("train:", train_df.shape, "test:", test_df.shape)
print(test_df["Class"].value_counts())
