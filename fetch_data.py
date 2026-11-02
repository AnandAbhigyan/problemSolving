"""Turns the raw UCI Dry Bean ARFF file into a plain CSV we can work with."""
import sys
from pathlib import Path

import pandas as pd

ARFF_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data_raw/DryBeanDataset/Dry_Bean_Dataset.arff")
OUT_PATH = Path("data/dry_bean.csv")


def read_arff(path):
    columns, rows, in_data = [], [], False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if low.startswith("@attribute"):
                columns.append(line.split()[1])
            elif low.startswith("@data"):
                in_data = True
            elif in_data:
                rows.append(line.split(","))
    df = pd.DataFrame(rows, columns=columns)
    feature_cols = [c for c in df.columns if c != "Class"]
    df[feature_cols] = df[feature_cols].astype(float)
    return df


if __name__ == "__main__":
    df = read_arff(ARFF_PATH)
    OUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH} -> {df.shape[0]} rows, {df.shape[1] - 1} features")
    print(df["Class"].value_counts())
