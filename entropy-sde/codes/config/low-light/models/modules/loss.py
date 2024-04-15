import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import sys
from .kde_utils import kde, cross_entropy, relative_entropy, hellinger, rule_of_thumb_h, create_window, spatial_entropy

class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target, weights=None):

        loss = self.loss_fn(predict, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss.mean()


class EntropyLoss(nn.Module):
    def __init__(self, pixel_level=64, name='relative'): # cross and relative entropy
        super(EntropyLoss, self).__init__()
        self.L = pixel_level
        self.name = name
        if name == 'learn':
            self.w = randn_weight(3,True)
            self.w.requires_grad = True

    def forward(self, x, y): 
        if self.name == 'relative':
            loss = r_entropy_loss(x, y, L=self.L)
        elif self.name == 'cross':
            loss = c_entropy_loss(x, y, L=self.L)
        elif self.name == 'hel':
            loss = hellinger_loss(x, y, L=self.L)
        elif self.name == 'helsmooth':
            loss = hellinger_loss(x, y, L=self.L) + smooth_loss(x, y, win_size=3)
        elif self.name == 'learn':
            loss = hellinger_loss_learnable(x, y, L=self.L,w = self.w)
        else:
            print('Choose relative or cross entropy!')

        return loss 


def randn_weight(size=3, gaussian=False):
    if not gaussian:
        return torch.randn(1, 1, 3, 3)
    weight = create_window(3, 1, 1.5, 1.5)
    idx = torch.randperm(weight.nelement())
    return weight.view(-1)[idx].view(weight.size())

def c_entropy_loss(input, target, L=64):
    p1 = kde(input, L) # [b, n, c, h, w]
    p2 = kde(target, L) # [b, n, c, h, w]
    return cross_entropy(p1, p2)

def r_entropy_loss(input, target, L=32, random_weight=True, auto_h=False):
    h = rule_of_thumb_h(target) if auto_h else 0.5
    w = randn_weight() if random_weight else None
    p1 = kde(input, L, weight=w, h=h) # [b, n, c, h, w]
    p2 = kde(target, L, weight=w, h=h) # [b, n, c, h, w]
    return relative_entropy(p2, p1)

def smooth_loss(input, target, win_size=3):
    weight = create_window(win_size, input.shape[1]).to(input.device)
    x = F.conv2d(input, weight, padding=win_size//2, groups=input.shape[1])
    y = F.conv2d(target, weight, padding=win_size//2, groups=input.shape[1])
    return (x - y).abs().mean()

def hellinger_loss(tensor1, tensor2, L=8, random_weight=True):
    w = randn_weight() if random_weight else None
    p1 = kde(tensor1, L, weight=w) # [b, n, c, h, w] 
    p2 = kde(tensor2, L, weight=w) # [b, n, c, h, w]
    return hellinger(p2, p1)

def hellinger_loss_learnable(tensor1, tensor2, L=8, w=None):
    p1 = kde(tensor1, L, weight=w) # [b, n, c, h, w] 
    p2 = kde(tensor2, L, weight=w) # [b, n, c, h, w]
    return hellinger(p2, p1)

