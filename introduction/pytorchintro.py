import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

x = torch.randn(3,requires_grad=True)
print(x)

## method 1
#x.requires_grad_(False)
#print(x)

## method 2
#y = x.detach()
#print(y)

## method 3
with torch.no_grad():
    y = x + 2
    print(y)