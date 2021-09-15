import torch

x = torch.Tensor([5, 3])
y = torch.Tensor([2, 1])

print(x*y)

z = torch.zeros([2, 5])

print(z)
print(z.shape)

z = torch.rand([2, 5])

z = z.view([1, 10])
print(z.shape)