## credit to python engineering youtube channel

import torch

# f = w*x
# f = 2*x

X = torch.tensor([1,2,3,4],dtype=torch.float32)
Y = torch.tensor([2,4,6,8],dtype=torch.float32)
W = torch.tensor([0.0],dtype=torch.float32,requires_grad=True)

#model prediction
def forward(x):
    return W*x

def loss(y,y_predicted):
    return ((y-y_predicted)**2).mean()

print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

# training 

lr = 0.01
n_iters = 20

for epoch in range(n_iters):
    # predict forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y,y_pred)

    # gradient = backward pass
    # dl/dw
    l.backward()

    # update waights
    # while updating wait you need to cancel gradient
    with torch.no_grad():
        W-= lr * W.grad

    # zero gradient
    ## _ means in place
    W.grad.zero_()

    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {W.item():.3f}, loss = {l.item():.8f}')


print(f'Prediction after training: f(5) = {forward(5).item():.3f}')



