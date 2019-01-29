import numpy as np

na = lambda arr: np.array(arr)
x = na([[3.,4.]]) # 1x2
y = na([[12.,16.]]) #1x2
w1 = na([[4.]]) #1x1
w3_2 = na([[.5]]) #1x1
w2 = na([[.1]]) #1x1
w3 = na([[.1]]) #1x1

def pred(x,y,w1,w2,w3):
    z1 = w1.dot(x) # 1x1 * 1x2 = 1x2
    z2 = w2.dot(z1) # 1x1 * 1x2 = 1x2
    z3 = w3.dot(z2) + z1 # 1x1 * 1x2 = 1x2
    dz3 = z3 - y
    return z1, z2, z3, dz3

m_fac = 1 / y.shape[1]
def res_grad(dz3, z2, z1, lr=0.05):
    dw3 = m_fac * lr * dz3.dot(z2.T) # 1x2 * 2x1 = 1x1
    dw3_2 = m_fac * lr * dz3.dot(z1.T) # 1x2 * 2x1 = 1x1
    return dw3, dw3_2

def grad(dz3, w1, w2, w3, z2, z1, x, lr=0.05):
    dz2 = w3.dot(dz3) # 1x1 * 1x2 = 1x2
    dz1 = w2.dot(dz2) # 1x1 * 1x2 = 1x2
    dw2 = m_fac * lr * dz2.dot(z1.T)
    dw1 = m_fac * lr * dz1.dot(x.T)
    return dw2, dw1

for i in range(100000):
    z1, z2, z3, dz3 = pred(x, y, w1, w2, w3)
    cost = np.sum(m_fac * np.square(dz3))
    # print(cost)
    dw3, dw3_2 = res_grad(dz3, z2, z2)
    dw2, dw1 = grad(dz3, w1, w2, w3, z2, z1, x)
    w3 -= dw3
    w3_2 -= dw3_2
    w2 -= dw2
    w1 -= dw1

# notice how w3 shrinks
print("w3 is ", w3)
print("w3_2 is ", w3_2)
print("w2 is ", w2)
print("w1 is ", w1)




