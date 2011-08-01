# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from BayesTLP import BayesTLP

#N = 30

## sample the function in two disjoint regions with N/2 samples each
#x = np.array([np.append(
	#np.random.normal(loc = 0.2, scale = 0.07, size=N/2),
	#np.random.normal(loc = 0.7, scale = 0.07, size=N/2)
	#)]).T
#xlin = np.array([np.arange(0,1,0.1)]).T

#eps = np.random.normal(loc = 0.0, scale = 0.05, size = x.shape) # Gaussian noise
#t = 0.5 + 0.4*np.sin(2*np.pi*x) + eps
#tlin = 0.5 + 0.4*np.sin(2*np.pi*xlin)

np.random.seed(0)

N = 16
noise = 0.1
x = np.array([0.25 + 0.07*np.random.normal(loc = 0.0, scale = 1, size = N)])
t = np.sin(2*np.pi*x) + noise * np.random.normal(loc = 0.0, scale = 1, size = N)

#x = np.loadtxt('nabneys_x')[None, :]
#t = np.loadtxt('nabneys_t')[None, :]

xlin = np.array([np.arange(0,1,0.005)]).T
tlin = np.sin(2*np.pi*xlin)

plt.figure()
plt.subplot(211)
plt.plot(xlin, tlin)
plt.scatter(x, t)
plt.title('Training set')

tlp = BayesTLP(H = 3, d = 1, ny = 1)

#tlp.w1 = np.loadtxt('nabneys_weights1', delimiter=',').T
#tlp.w2 = np.loadtxt('nabneys_weights2', delimiter=',')[None, :]
#tlp.w1 = np.loadtxt('nabneys_initial_weights1', delimiter=',').T
#tlp.w2 = np.loadtxt('nabneys_initial_weights2', delimiter=',')[None, :]

tlp.train_bayes(x.T, t.T, alpha = 0.01, beta = 50, nouter = 1, ninner = 1, Nmax = 500, gtol=1e-11)

ylin = tlp.forward(xlin)
plt.subplot(212)
plt.plot(xlin,ylin)
plt.title('Network output')

# get error bars
XLIN = np.append(np.ones([xlin.shape[0],1]),xlin,1)
X = np.append(np.ones([x.T.shape[0],1]),x.T,1)
g1, g2 = tlp.deriv(XLIN)
g1 = np.reshape(g1, [xlin.shape[0], tlp.ny, tlp.w1.size])
g2 = np.reshape(g2, [xlin.shape[0], tlp.ny, tlp.w2.size])
g = np.append(g1, g2, axis = 2)

y = tlp.forward(x.T)
A = tlp.HE_bayes(X, y, t, tlp.w1, tlp.w2)

sigma_t = np.zeros([XLIN.shape[0]])
for nt in range(XLIN.shape[0]):
  sigma_t[nt] = np.sqrt(1.0/tlp.beta + np.dot(g[nt, 0].T, np.dot(np.linalg.inv(A), g[nt, 0])))

# plot error bars
plt.plot(xlin, ylin[:,0] + sigma_t, color='r')
plt.plot(xlin, ylin[:,0] - sigma_t, color='r')

plt.show()


