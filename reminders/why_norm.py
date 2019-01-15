import numpy as np

def gd(w, x, y, lr):
    return lr * (w.dot(x) - y).dot(x.T)



w = np.array([[0.1,0.2]])
x = np.array([[100, 50], [1, 2]])
y = np.array([[102, 54]])


# the problem here is the zig zags w/ the gradients
# so really the cost, since it's a linear func
# that's because the big input's gradient is amplified too much
# and so it zig zags to find the correct weight confs instead of 
# converging in a linear like manner, just like ng said
hmm = 0
k = 0
for i in range(20010):
    gd1 = gd(w, x, y, 0.0001)
    pred = w.dot(x)
    w -= gd1
    if (w[0][1] < k):
        print(i, "-----------------------------------", w[0][1])
    k = w[0][1]
    if (i > 20000 or i < 10):
        print(pred, w, gd1)


print('-------------------')
print(w)
print(w.dot(x))
