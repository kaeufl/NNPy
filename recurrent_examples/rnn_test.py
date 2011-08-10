import numpy as np
from rnn import RNN

net = RNN(H = 3, d = 1, ny = 1, T = 3)

x = np.array([[[0], [0.5], [1]]])
t = np.array([[[0], [0.5], [1]]])

net.train_BFGS(x, t, gtol=1e-5, Nmax=200)

print net.forward(x)