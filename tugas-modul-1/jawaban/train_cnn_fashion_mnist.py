import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import os
import random

# untuk mengatur reproducibility
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transformasi penting, normalisasi gambar jadi float
transform = transforms.Compose([
    transforms.ToTensor(),  # [0,255] -> [0,1]
])

# Download Fashion-MNIST
train_dataset = datasets.FashionMNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, transform=transform, download=True
)

# train val split, rasionya bebas, di sini 90-10
val_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model sederhana saja
class CNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x

model = CNN().to(device)

# loss dan optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# training loop + validasi setiap epoch
n_epochs = 10
save_path = "fashion_cnn_torchvision.pth"
best_val_loss = float("inf")

for epoch in range(1, n_epochs + 1):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # Validation
    model.eval()
    val_losses = []
    val_preds = []
    val_trues = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_losses.append(loss.item())
            val_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            val_trues.append(yb.cpu().numpy())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    val_preds = np.concatenate(val_preds)
    val_trues = np.concatenate(val_trues)
    val_acc = accuracy_score(val_trues, val_preds)

    # checkpointing
    # logika sederhana, cuma menyimpan model terakhir yang lebih rendah lossnya dari yang sebelumnya tersimpan
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)

    print(f"Epoch {epoch:2d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

# load model terbaik buat testing
best_model = CNN().to(device)
best_model.load_state_dict(torch.load(save_path, map_location=device))
best_model.eval()

test_preds = []
test_trues = []
# untuk mempercepat komputasi, testing tidak perlu menghitung gradien
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = best_model(xb)
        test_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        # karena tanpa gradien, tidak perlu .detach()
        test_trues.append(yb.cpu().numpy())

test_preds = np.concatenate(test_preds)
test_trues = np.concatenate(test_trues)
test_acc = accuracy_score(test_trues, test_preds)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Best model saved to: {os.path.abspath(save_path)}")
