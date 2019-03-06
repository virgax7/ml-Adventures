import numpy as np

# column vectors as inputs
x = np.array([[1, 1, .5],
              [1, 1, 1]])

w = np.array([
    [.5, .5]
])

y = np.array([[1, 1, 0]])


for i in range(10000):
    pred = w.dot(x)
    print("w is ", w)
    print("loss is ", np.sum((pred - y) ** 2))
    print("pred is ", pred)

    # descent
    grad = 1 / y.shape[0] * 2 * (pred - y).dot(x.T)
    print("pure gradient is ", grad)
    w -= .1 * grad
    print("\n ------------ \n")

# the important outputs can be seen immediately
#
# w is  [[0.5 0.5]]
# loss is  0.5625
# pred is  [[1.   1.   0.75]]
# pure gradient is  [[0.75 1.5 ]]
# ------------
# w is  [[0.425 0.35 ]]
# loss is  0.41765625000000006
# pred is  [[0.775  0.775  0.5625]]
# pure gradient is  [[-0.3375  0.225 ]]
# ------------
# w is  [[0.45875 0.3275 ]]
# loss is  0.40148789062499995
# pred is  [[0.78625  0.78625  0.556875]]
# pure gradient is  [[-0.298125  0.25875 ]]
# ------------
# w is  [[0.4885625 0.301625 ]]
# loss is  0.3860562041015625
# pred is  [[0.7901875  0.7901875  0.54590625]]
# pure gradient is  [[-0.29334375  0.2525625 ]]
# ------------

# this shows the first weight .5 of [[0.5,0.5]] first soaking up the error and becoming .425
# but the effect is more dramatic for the second .5 because the input 1 is stronger than input .5
# anyhow this error gets alleviated as the second weight now is in a position where the first can grow again
# and it repeats this kind of pattern until

# after 10000 runs
# w is  [[ 2. -1.]]
# loss is  3.303355040612987e-30
# pred is  [[1.00000000e+00 1.00000000e+00 1.55431223e-15]]
# pure gradient is  [[-1.11022302e-15  4.44089210e-16]]



