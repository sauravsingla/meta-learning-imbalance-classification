"""
Extended Benchmarking with Multiple Imbalanced Datasets + W&B Logging
"""

import wandb
import torch
from train_meta import MetaLearner, train_meta_learner, evaluate
from sklearn.datasets import make_classification, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def load_openml_dataset(name, n_samples=5000):
    data = fetch_openml(name, version=1, as_frame=False)
    X, y = data.data, data.target
    if isinstance(y[0], str):  # Convert string labels to binary
        classes = list(set(y))
        y = np.array([1 if val == classes[1] else 0 for val in y])
    X, y = X[:n_samples], y[:n_samples]
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

def prepare_dataloader(X_train, y_train):
    tensor_x = torch.tensor(X_train, dtype=torch.float32)
    tensor_y = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=64, shuffle=True)

def run_benchmark_wandb():
    wandb.init(project="meta-learning-imbalance-benchmark", config={"epochs": 20})

    datasets = {
        "synthetic": make_classification(n_samples=5000, n_features=20, weights=[0.99, 0.01], random_state=1),
        "creditcard": load_openml_dataset("credit-g"),
        "kddcup": load_openml_dataset("KDDCup99")
    }

    for name, data in datasets.items():
        print(f"Running benchmark on: {name}")
        if name == "synthetic":
            X, y = data
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
        else:
            X_train, X_test, y_train, y_test = data

        loader = prepare_dataloader(X_train, y_train)
        model = MetaLearner(input_dim=X_train.shape[1], hidden_dim=64).to("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(wandb.config["epochs"]):
            train_meta_learner(model, loader, optimizer, "cpu")

        f1, auc = evaluate(model, X_test, y_test, "cpu")
        print(f"{name} - F1: {f1:.4f}, AUC: {auc:.4f}")
        wandb.log({f"{name}_f1": f1, f"{name}_auc": auc})

    wandb.finish()

if __name__ == "__main__":
    run_benchmark_wandb()
