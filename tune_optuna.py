"""
Auto-tune hyperparameters using Optuna for the meta-learner.
"""

import optuna
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from train_meta import MetaLearner, train_meta_learner, evaluate

def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    X, y = make_classification(n_samples=5000, n_features=20, weights=[0.99, 0.01], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    ), batch_size=batch_size, shuffle=True)

    model = MetaLearner(20, hidden_dim).to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(10):
        train_meta_learner(model, train_loader, optimizer, "cpu")

    f1, _ = evaluate(model, X_test, y_test, "cpu")
    return f1

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best trial:", study.best_trial)
