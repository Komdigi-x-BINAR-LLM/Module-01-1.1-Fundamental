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

# atur randomness semua library yang dipakai
# TODO: implemen ini


# download dataset iris lewat scikit learn
iris = load_iris()
X = iris.data.astype(np.float32)        # shape (150, 4)
y = iris.target.astype(np.int64)        # 0,1,2


# TODO: train val split dataset di atas


# TODO: implemen preprocessing, scaling aja cukup


# TODO: implemen dataloader
# hint: datasetnya masih berbentuk array numpy, 
# ubah dulu jadi torch dataset. Kelas dataset mana yang cocok untuk array numpy?
# cek di sini https://docs.pytorch.org/docs/stable/data.html


# TODO: implemen model
class LogisticRegression(nn.Module):
    ...

# TODO: loss dan optimizer

# TODO: pilih optimizer dan loss function, lalu implementasi loop training 
# + menghitung validation loss setiap epoch

# TODO implementasi evaluasi, pakai saja metric klasifikasi yang umum