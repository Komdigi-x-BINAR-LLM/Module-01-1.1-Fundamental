import torch
from torch import nn

class LogisticRegression(nn.Module):
    def __init__(self, input_features, class_num):
        super().__init__()
        self._linear_layer = nn.Linear(5, 2, bias=True)
        self._sigmoid = nn.Sigmoid()
        
    def forward(self, X):
        return self._sigmoid(self._linear_layer(X))

log_reg_module = LogisticRegression(8, 2)

# menyimpan
torch.save(log_reg_module.state_dict(), "model.pt")

# memuat kembali parameter ke model
# model harus sama konfigurasinya
load_model = LogisticRegression(8, 2)
state_dict = torch.load("model.pt")
load_model.load_state_dict(state_dict)


# contoh lain, menyimpan seluruh kebutuhan training
# misal menyimpan state dari optimizer juga

optimizer = torch.optim.SGD(log_reg_module.parameters(), lr=0.001, momentum=0.005)
# yang disimpan dan diabaca oleh torch.save 
# sebenarnya dictionary python biasa,
# jadi cukup fleksibel
torch.save({
    'model_state_dict': log_reg_module.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    # bisa juga ditambahkan informasi lainnya asal dalam bentuk dict
    'date': "10-10-2025",
    "total_epoch": 10
}, "train_checkpoint.pt")

# load lagi
checkpoint_dict = torch.load("train_checkpoint.pt")
model_state_dict = checkpoint_dict["model_state_dict"]
optimizer_state_dict = checkpoint_dict["optimizer_state_dict"]
train_date = checkpoint_dict["date"]
train_total_epochs = checkpoint_dict["total_epoch"]

loaded_model = LogisticRegression(8, 2)
loaded_optimizer = torch.optim.SGD(log_reg_module.parameters(), lr=0.001, momentum=0.005)
loaded_model.load_state_dict(model_state_dict)
loaded_optimizer.load_state_dict(optimizer_state_dict)
