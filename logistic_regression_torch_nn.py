import torch
from torch import nn

# inputnya random dulu, untuk ilustrasi
# anggap saja ada 8 input
X = torch.randn(size=(8, 5), dtype=torch.float32)

linear_layer = nn.Linear(5, 2, bias=True)
sigmoid = nn.Sigmoid()

y = sigmoid(linear_layer(X))
y = y.max(dim=1)

print(y)
# Contoh Output (akan berubah karena random):
# torch.return_types.max(
# values=tensor([0.8213, 0.5836, 0.6810, 0.7051, 0.7227, 0.6645, 0.5628, 0.7392],
#        grad_fn=<MaxBackward0>),
# indices=tensor([1, 0, 0, 1, 1, 0, 0, 1]))