import torch
from torch import nn

def tan_square(input, target, reduction='mean'):
    assert reduction in ['mean', 'sum'], \
        f"Expected reduction to be either 'mean' or 'sum', got {reduction}."
    temp = torch.tan((input-target) * torch.pi / 2)**2
    return torch.mean(temp) if reduction == 'mean' else torch.sum(temp)

def log_product(input, target, reduction='mean'):
    assert reduction in ['mean', 'sum'], \
        f"Expected reduction to be either 'mean' or 'sum', got {reduction}."
    diff = input - target
    temp = - torch.log(1 + diff) * torch.log(1 - diff)
    return torch.mean(temp) if reduction == 'mean' else torch.sum(temp)