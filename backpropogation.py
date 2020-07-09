import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0,requires_grad=True)

# forward pass
y_hat = w * x
loss = (y_hat - y)**2
print(loss)
#backward
loss.backward()
print(w.grad)
