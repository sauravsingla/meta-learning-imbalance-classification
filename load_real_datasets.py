"""
Integration with real-world imbalanced datasets (MIMIC-III, CIC-IDS).
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_cic_ids(file_path):
    df = pd.read_csv(file_path)
    df = df.select_dtypes(include=["number"])
    df = df.dropna()
    X = df.drop("Label", axis=1, errors="ignore")
    y = df["Label"] if "Label" in df.columns else (df.iloc[:, -1] > 0).astype(int)
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.2, stratify=y)

def load_mimic_iii(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    y = df["mortality"] if "mortality" in df.columns else (df.iloc[:, -1] > 0).astype(int)
    X = df.drop("mortality", axis=1, errors="ignore")
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.2, stratify=y)

if __name__ == "__main__":
    print("This module is for loading MIMIC-III and CIC-IDS datasets.")
    print("Use `load_cic_ids(file_path)` or `load_mimic_iii(file_path)` inside your training pipeline.")
