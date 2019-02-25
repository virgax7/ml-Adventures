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


def softmax(z):
    return np.exp(z) / sum(np.exp(z))

first_pred = softmax(w.dot(x))
print(first_pred)

def loss_vector(z, y):
    return -y * np.log(softmax(z))

print(loss_vector(first_pred, y))

def softmax_gd(w, x, y):
    # loss is -ylog(pred)
    ewx = np.exp(w.dot(x))
    # da = -y * (ewx * (sum(ewx) - ewx) / sum(ewx) ** 2) / (ewx / sum(ewx)) + ((1 - y) * (ewx * (sum(ewx) - ewx) / sum(ewx) ** 2) / (1 - (ewx / sum(ewx))))
    da = (1 - y) * (ewx / sum(ewx))  - y * (1 - (ewx / sum(ewx)))
    dw = (da).dot(x.T)
    return dw

print("------------- now training----------------------------\n\n\n")
cycles = 500
for i in range(cycles):
    z = w.dot(x)
    # print("loss is ", sum(loss_vector(z, y)))
    if (i == 0):
        print("notice how the gradient value is the same as softmax_numpy.py ")
        print("gradient value for the first gradient descent is \n ",softmax_gd(w, x, y))
    w -= 0.1 * softmax_gd(w, x, y)

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print("first pred was \n ", first_pred)
print("after 500 gradient descent, pred is now \n ", softmax(w.dot(x)))
print("the weights are \n", w)

