import torch
from torch import nn
from sklearn.metrics import accuracy_score
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, gamma=0):
        super().__init__()
        self._gamma = gamma

    def forward(self, predicted_probabilities, ground_truth, reduction="mean"):
        gt_probs = predicted_probabilities[
            torch.arange(len(ground_truth)), ground_truth
        ]
        focal_gain = torch.pow((1 - gt_probs), self._gamma) * -1
        log_loss = torch.log(gt_probs)
        loss = focal_gain * log_loss
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss


test_input = torch.softmax(torch.randn((32, 10), dtype=torch.float32), dim=1)
test_gt = torch.randint(0, 10, (32,), dtype=torch.int)

loss_fn = FocalLoss()

loss_val = loss_fn(test_input, test_gt)

print(loss_val)
print(loss_val.shape)


# membuat lagi contoh logistic regression untuk dihitung gradiennya

W = torch.randn(size=(64, 4), dtype=torch.float32, requires_grad=True)
B = torch.zeros(size=(1, 4), dtype=torch.float32, requires_grad=True)

X = torch.randn(size=(16, 64), dtype=torch.float32)
y = torch.randint(0, 4, (16,), dtype=torch.int)

# inisialisasi optimizer setelah model selesai didefinisikan
optim = torch.optim.SGD(params=[W, B], lr=1e-3)

Y_pred = torch.softmax(torch.sigmoid(torch.matmul(X, W) + B), dim=1)
loss = loss_fn(Y_pred, y)
# best practice sebelum menghitung gradien adalah 
# membersihkan gradien yang sebelumnya sudah dihitung
optim.zero_grad()
loss.backward()

print(W.grad)
# Output
# tensor([[ 1.0020e-03,  8.7977e-03,  5.8582e-03, -4.4702e-03],
#                               ...
#         [-4.0570e-03,  2.4681e-03,  2.7378e-03, -7.1247e-03]])
print(W.grad.shape)
# Output
# torch.Size([64, 4])

print(f"Loss before weight update: {loss}")
# Output
# Loss before weight update: 1.4773516654968262
optim.step() # melakukan update parameter
# menghitung kembali loss
Y_pred = torch.softmax(torch.sigmoid(torch.matmul(X, W) + B), dim=1)
loss = loss_fn(Y_pred, y)
print(f"Loss after weight update: {loss}")
# Output
# Loss after weight update: 1.4773342609405518

y_numpy = y.detach().numpy().astype(int)
preds_numpy = Y_pred.detach().numpy().astype(int)

class_preds_numpy = np.max(preds_numpy, axis=1)

print(accuracy_score(y_numpy, class_preds_numpy))
# Output
# 0.0625 (bisa berbeda karena random)
