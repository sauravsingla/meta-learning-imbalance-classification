import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class MetaLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def create_imbalanced_data(n_samples=10000, weights=[0.995, 0.005]):
    X, y = make_classification(
        n_samples=n_samples, n_features=20, n_informative=10, 
        weights=weights, flip_y=0.01, random_state=42
    )
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def train_meta_learner(model, data_loader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(X_test).cpu().numpy()
        preds = np.argmax(logits, axis=1)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, logits[:, 1])
        return f1, auc

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_test, y_train, y_test = create_imbalanced_data()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = MetaLearner(input_dim=20, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in tqdm(range(20), desc="Training Meta-Learner"):
        train_meta_learner(model, loader, optimizer, device)

    f1, auc = evaluate(model, X_test, y_test, device)
    print(f"F1 Score: {f1:.4f}, AUC: {auc:.4f}")

if __name__ == "__main__":
    run()
