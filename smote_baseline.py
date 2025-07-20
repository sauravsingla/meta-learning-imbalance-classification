"""
Baseline using SMOTE + simple neural net classifier.
"""

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.datasets import make_classification
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def run_smote_baseline():
    X, y = make_classification(n_samples=5000, weights=[0.99, 0.01], random_state=42)
    X_res, y_res = SMOTE().fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    model = SimpleNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(10):
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        preds = logits.argmax(axis=1)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, logits[:, 1])
        print(f"SMOTE Baseline - F1: {f1:.4f}, AUC: {auc:.4f}")

if __name__ == "__main__":
    run_smote_baseline()
