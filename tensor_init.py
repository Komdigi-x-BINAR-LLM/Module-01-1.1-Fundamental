import torch
import numpy as np

literal_matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

np_array = np.array(literal_matrix)

manual_torch_tensor = torch.Tensor(literal_matrix)
torch_tensor_from_numpy = torch.from_numpy(np_array)


a = torch.zeros(2, 3)
print(a)
b = torch.ones_like(a)
print(b)
c = torch.arange(0, 10, step=2)
print(c)
d = torch.randn(3, 3)
print(d)
