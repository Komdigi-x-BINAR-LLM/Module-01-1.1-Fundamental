import torch

# input memiliki dimensi 5, output memiliki dimensi 2 (2 kelas)

# inisialisasi weights sebagai random
W = torch.randn(size=(5, 2), dtype=torch.float32)
# inisialisasi bias dengan nilai awal nol
B = torch.zeros(size=(1, 2), dtype=torch.float32)

# inputnya random dulu, untuk ilustrasi
# anggap saja ada 8 input
X = torch.randn(size=(8, 5), dtype=torch.float32)

# transformasi linear
y = torch.matmul(X, W) + B
# sigmoid
y = 1 / (1 + torch.exp(-1 * y))
# reduksi, menghitung indeks kelas mana yang paling besar skor confidencenya
# dim = 1 artinya maksimum dihitung dari seluruh anggota dimensi ke 1
# dalam kasus ini, menghitung maksimum dari skor confidence setiap kelas untuk
# setiap baris sampel
y = y.max(dim=1)

print(y)
# Contoh Output (akan berubah karena random):
# torch.return_types.max(
# values=tensor([0.8953, 0.7578, 0.2555, 0.8783, 0.5772, 0.8984, 0.9424, 0.8327]),
# indices=tensor([0, 1, 0, 1, 0, 0, 1, 0]))