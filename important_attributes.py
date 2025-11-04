import torch

print(torch.cuda.is_available())
# Output: False

t1 = torch.Tensor([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print(t1)
# Output:
# tensor([[ 1.,  2.,  3.,  4.],
#         [ 5.,  6.,  7.,  8.],
#         [ 9., 10., 11., 12.]])

print(t1.shape)
# Output: torch.Size([3, 4])

print(t1.device)
# Output: device(type='cpu')

# Skip the following if GPU is not available
if torch.cuda.is_available():
    t1_cuda = t1.to('cuda')
    print(t1_cuda.device)
    # Output: device(type='cuda', index=0)

print(t1.dtype)
# Output: torch.float32
