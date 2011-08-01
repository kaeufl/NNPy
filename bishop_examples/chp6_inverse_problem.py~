# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mdn import MDN
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from datetime import datetime

np.seterr(invalid='raise')

x = np.array([np.arange(0,1,1.0/300)]).T # x for plotting

m = np.array([np.arange(0,1,1.0/300)]).T
eps = np.random.uniform(-0.1,0.1,x.shape) # uniform noise
d = (m + 0.3*np.sin(2*np.pi*m) + eps) # sin (x) + noise

#################################
#train on the inverse problem
# input:  data d
# target: model m
#################################
M = 3
mdn = MDN(H=5, d = 1, ny = 1, M = M)

# init weights to model the unconditional p(d)
mdn.init_weights(m, 100)

## now the network should approximate the distribution of the target data
#plt.figure()
#plt.hist(m,50, normed=True)
#y = mdn.forward(x)

## make a plot
#c = mdn.ny / (3*M)
#alpha = y[:, 0:M]
#sigma = y[:, M:2*M]
#mu = y[:, 2*M:]
#phi = np.zeros([M, x.shape[0]])
#prior = np.zeros([x.shape[0]])
#for k in range(M):
  #phi[k, :] = (1.0 / (2*np.pi*sigma[:,k])**(0.5)) * np.exp(- 1.0 * (x[:,0] - mu[:,k])**2 / (2 * sigma[:,k]))
  #prior = prior + alpha[:,k] * phi[k, :]
#plt.figure()
#plt.plot(x, prior)
#plt.xlim([0,1])
#plt.ylim([0,1])
#plt.show()

mdn.train_BFGS(d, m, 0.5, 400)

y = mdn.forward(x)
c = mdn.ny / (3*M)
alpha = y[:, 0:M]
sigma = y[:, M:2*M]
mu = y[:, 2*M:]

plt.figure()
plt.plot(x, alpha[:,0])
plt.plot(x, alpha[:,1])
plt.plot(x, alpha[:,2])
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('mixing coefficients')
plt.figure()
plt.plot(x, mu[:,0])
plt.plot(x, mu[:,1])
plt.plot(x, mu[:,2])
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('means')

# make a contour plot
k = 0
#X = x[0,:]
[X,Y]=np.meshgrid(x,x)

#T = t[0,:]
nx = x.shape[0]
nt = m.shape[0]
phi = np.zeros([M,nx, nt])
P_cond = np.zeros([nx, nt])

for k in range(M):
  SIGMA = np.tile(sigma[:,k],[nx, 1]).T
  MU = np.tile(mu[:,k],[nx, 1]).T
  phi[k,:,:] = (1.0 / (2*np.pi*SIGMA)**(0.5)) * np.exp(- 1.0 * (X-MU)**2 / (2 * SIGMA))
  P_cond[:,:] = P_cond[:,:] + phi[k,:,:] * alpha[:,k]

f = plt.figure()
ax = Axes3D(f)
ax.plot_surface(X,Y,P_cond, rstride=10, cstride=10, cmap=cm.jet, linewidth=0, antialiased=False)
#plt.contour(X,Y,P_cond)

#show error
#plt.figure()
#plt.plot(mdn.E)

plt.show()

