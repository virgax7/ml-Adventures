import torch
from torch.autograd import Variable
import numpy as np

x = np.array([
    [2.],
    [3.]
])
w = np.array([
    [.1, .2],
    [.3, .3],
    [.2, .1]
])
# label encoded
y = np.array([
    [2]
])

def one_hot(y, n):
    return np.array([[1 if i == j[0] else 0 for i in range(n)] for j in y]).T
# one hot encoded
y = one_hot(y, 3)

x = torch.from_numpy(x).double()
y = torch.from_numpy(y).double()
w = Variable(torch.from_numpy(w).double(), requires_grad=True)


def softmax(z):
    return torch.div(torch.exp(z) , torch.sum(torch.exp(z)))

def loss_vector(x, w, y):
    return torch.sum(-y * torch.log(softmax(w.mm(x))))


first_pred = softmax(w.mm(x))

cycles = 1
for i in range(cycles):
    l = loss_vector(x, w, y)
    l.backward()
    if (i == 0):
        print("notice how the gradient value is the same as softmax_numpy.py ")
        print("gradient value for the first gradient descent is \n", w.grad.data)
    w.data -= 0.1 * w.grad.data
    w.grad.data.zero_()

# torch.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print("\n\n-----------------------\n\n")
torch.set_printoptions(precision=2)
print("first pred was \n ", first_pred)
print("after 500 gradient descent, pred is now \n ", softmax(w.mm(x)))
print("the weights are \n", w)
