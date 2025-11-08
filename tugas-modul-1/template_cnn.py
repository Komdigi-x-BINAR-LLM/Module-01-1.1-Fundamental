import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import os
import random

# atur randomness semua library yang dipakai
# TODO: implemen ini

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: implementasi preprocess gambar
transform = ...

# Download Fashion-MNIST
# ini sudah dalam bentuk objek Dataset, bisa langsung dipakai untuk 
# inisialisasi dataloader
train_dataset = datasets.FashionMNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, transform=transform, download=True
)
# TODO: implementasi dataloader untuk semua data

train_loader = ...
val_loader = ...
test_loader = ...

# TODO: Implementasi model, gunakan module

class CNN(nn.Module):
    ...


# TODO: pilih optimizer dan loss function, lalu implementasi loop training 
# + menghitung validation loss setiap epoch

# TODO implementasi evaluasi, pakai saja metric klasifikasi yang umum