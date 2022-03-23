import torch
import torch.nn as nn

a = torch.tensor([[1, 2, 3]).float()
b = torch.tensor([1, 0, 1]).float()

print(nn.CrossEntropyLoss()(a, b))