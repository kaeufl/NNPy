import numpy as np
import matplotlib.pyplot as plt
from rnn import RNN

N = 1
T = 50

#------------------------------------------------------------------------------
# TODO: 
# - add noise (regression problem)
#
#------------------------------------------------------------------------------ 

def generate_data(a, b, T, N = 1, n = 0):
    def f(x):
        if n != 0:
            eps = np.random.normal(0, n, size = x.shape)
        else:
            eps = 0
        return np.sin(a * x) + b + eps
        
    if N > 1:
        x = np.random.uniform(0, 1, size = [N, T])
        y = f(x)
        return x[:,:,None], y[:,:,None]
    else:
        x = np.linspace(0, 1, T)
        y = f(x)
        return x[None,:,None], y[None,:,None]

plt.figure()

d, t = generate_data(np.pi, 0.0, T, N)

net = RNN(H = 1, d = 1, ny = 1, T = T)
net.init_weights(1e2, 1e2)
net.train_BFGS(d, t, gtol=1e-12, Nmax=200, constrained = False)

plt.subplot(2,2,1)
plt.title("Training data (squares) \n and network prediction (red line), N = "+str(N)+" T = " + str(T))
plt.scatter(d[:,:,0],t[:,:,0],s=20, marker='s')

x = np.linspace(0,1,T)
y = net.forward(x[None,:,None])
plt.plot(x, y[0,:,0], color='r')

plt.subplot(2,2,2)
plt.title('Test')
x = np.linspace(0,1,T)
y = net.forward(x[None,:,None])
plt.plot(x, y[0,:,0])

plt.subplot(2,2,4)
plt.text(0,0.7,"Network is trained on sequences of \n "+str(T)+" time-steps in the interval [0,1]")
plt.axis('off')

plt.suptitle('Approximation of a time-series using a recurrent neural network.')
plt.show()

# net.train_BFGS(x, t, gtol=1e-5, Nmax=200)

#print net.forward(x)