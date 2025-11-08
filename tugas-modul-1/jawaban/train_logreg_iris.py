import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
import random

# biar reproducible
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


# download dataset iris lewat scikit learn
iris = load_iris()
X = iris.data.astype(np.float32)        # shape (150, 4)
y = iris.target.astype(np.int64)        # 0,1,2


# splitting 60/20/20
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=seed
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=seed
)  # 0.25 x 0.8 = 0.2

# preprocess sederhana
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
X_test = scaler.fit_transform(X_test)

# dataloader
batch_size = 16
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# CrossEntropyLoss langsung dihitung pada logits
# jadi tidak perlu sigmoid dan softmax
class LogisticRegression(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.linear(x)

model = LogisticRegression(in_features=X.shape[1], n_classes=len(np.unique(y)))

# loss dan optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# training loop
n_epochs = 100
best_val_loss = float("inf")
save_path = "iris_logistic_state.pth"

for epoch in range(1, n_epochs + 1):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb = xb
        yb = yb
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # Validation
    model.eval()
    val_losses = []
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb
            yb = yb
            logits = model(xb)
            loss = criterion(logits, yb)
            val_losses.append(loss.item())
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            trues.append(yb.cpu().numpy())

    val_loss = float(np.mean(val_losses))
    train_loss = float(np.mean(train_losses))
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    val_acc = accuracy_score(trues, preds)

    # checkpointing
    # logika sederhana, cuma menyimpan model terakhir yang lebih rendah lossnya dari yang sebelumnya tersimpan
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

# load model terakhir (loss paling rendah di val set) lalu eval
best_model = LogisticRegression(in_features=X.shape[1], n_classes=len(np.unique(y)))
best_model.load_state_dict(torch.load(save_path))
best_model.eval()

test_preds = []
test_trues = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb
        logits = best_model(xb)
        test_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        test_trues.append(yb.numpy())

test_preds = np.concatenate(test_preds)
test_trues = np.concatenate(test_trues)
test_acc = accuracy_score(test_trues, test_preds)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Saved model state to: {os.path.abspath(save_path)}")
