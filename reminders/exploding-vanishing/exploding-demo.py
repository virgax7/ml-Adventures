import torch
from torch.autograd import Variable

w1 = Variable(torch.Tensor([[-1, 2], [2, -1]]), requires_grad=True)
w2 = Variable(torch.Tensor([[-1, 2], [2, -1]]), requires_grad=True)
w3 = Variable(torch.Tensor([[-1, 2], [2, -1]]), requires_grad=True)
w4 = Variable(torch.Tensor([[-1, 2], [2, -1]]), requires_grad=True)
w5 = Variable(torch.Tensor([[-1, 2], [2, -1]]), requires_grad=True)
w6 = Variable(torch.Tensor([[-1, 2], [2, -1]]), requires_grad=True)
w7 = Variable(torch.Tensor([[-1, 2], [2, -1]]), requires_grad=True)
w8 = Variable(torch.Tensor([[-1, 2], [2, -1]]), requires_grad=True)
w9 = Variable(torch.Tensor([[2, -1]]), requires_grad=True)

def loss(y):
    return (y - 2) ** 2

for i in range(3):
    x = torch.Tensor([[1], [1]])
    y = w9.mm(w8.mm(w7.mm(w6.mm(w5.mm(w4.mm(w3.mm(w2.mm(w1.mm(x)))))))))
    print("y is ", y)
    l = loss(y)
    l.backward()
    print("loss is ", l)
    print(w1.grad.data)
    print(w2.grad.data)
    print(w3.grad.data)
    print(w4.grad.data)
    print(w5.grad.data)
    print(w6.grad.data)
    print(w7.grad.data)
    print(w8.grad.data)
    print(w9.grad.data)
    # you can tune the learning rate very low e.g. .00000001 but then you have other problems in real world problems..
    # gradient clipping would be better, but still exploding gradients are a real mess
    w1.data -= .00001 * w1.grad.data
    w2.data -= .00001 * w2.grad.data
    w3.data -= .00001 * w3.grad.data
    w4.data -= .00001 * w4.grad.data
    w5.data -= .00001 * w5.grad.data
    w6.data -= .00001 * w6.grad.data
    w7.data -= .00001 * w7.grad.data
    w8.data -= .00001 * w8.grad.data
    w9.data -= .00001 * w9.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()
    w3.grad.data.zero_()
    w4.grad.data.zero_()
    w5.grad.data.zero_()
    w6.grad.data.zero_()
    w7.grad.data.zero_()
    w8.grad.data.zero_()
    w9.grad.data.zero_()
    print("-----------------")







==================================output

y is  tensor([[1.]], grad_fn=<MmBackward>)
loss is  tensor([[1.]], grad_fn=<PowBackward0>)
tensor([[ 6560.,  6560.],
        [-6562., -6562.]])
tensor([[-2188., -2188.],
        [ 2186.,  2186.]])
tensor([[ 728.,  728.],
        [-730., -730.]])
tensor([[-244., -244.],
        [ 242.,  242.]])
tensor([[ 80.,  80.],
        [-82., -82.]])
tensor([[-28., -28.],
        [ 26.,  26.]])
tensor([[  8.,   8.],
        [-10., -10.]])
tensor([[-4., -4.],
        [ 2.,  2.]])
tensor([[-2., -2.]])
-----------------
y is  tensor([[969.5535]], grad_fn=<MmBackward>)
loss is  tensor([[936159.6875]], grad_fn=<PowBackward0>)
tensor([[-6243025.5000, -6243025.5000],
        [ 6453211.0000,  6453211.0000]])
tensor([[ 1849307.1250,  2407930.7500],
        [-1827522.8750, -2379566.0000]])
tensor([[-1010655.0000,  -395587.8438],
        [ 1017132.4375,   398123.2188]])
tensor([[ -77180.4531,  549660.6875],
        [  76454.8203, -544492.9375]])
tensor([[-385803.1875,  231014.1562],
        [ 395609.2188, -236885.8750]])
tensor([[-296865.4375,  351057.4688],
        [ 275622.3438, -325936.5000]])
tensor([[-285399.5938,  269917.3438],
        [ 356768.7500, -337414.9062]])
tensor([[-412634.2188,  420375.8438],
        [ 206310.9062, -210181.6094]])
tensor([[ 626686.8125, -622815.9375]])
-----------------
y is  tensor([[-4007362.2500]], grad_fn=<MmBackward>)
loss is  tensor([[1.6059e+13]], grad_fn=<PowBackward0>)
tensor([[-1.5304e+12, -1.5304e+12],
        [-1.7551e+12, -1.7551e+12]])
tensor([[ 2.6387e+13, -2.6849e+13],
        [ 1.5867e+13, -1.6145e+13]])
tensor([[-1.5343e+12,  1.5068e+12],
        [-1.1319e+13,  1.1116e+13]])
tensor([[ 7.0357e+12, -7.0831e+12],
        [-2.7649e+12,  2.7835e+12]])
tensor([[ 4.2133e+12, -4.1460e+12],
        [-5.6888e+12,  5.5980e+12]])
tensor([[ 4.9902e+12, -5.2109e+12],
        [-4.6293e+12,  4.8340e+12]])
tensor([[ 4.9253e+12, -4.3273e+12],
        [-5.4634e+12,  4.8001e+12]])
tensor([[ 3.7946e+12, -5.6169e+12],
        [-4.6496e+12,  6.8824e+12]])
tensor([[-5.6813e+12,  1.5065e+12]])
