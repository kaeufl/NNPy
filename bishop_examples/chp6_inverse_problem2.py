# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mlp import TLP, MDN
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

np.seterr(invalid='raise')

tlp = TLP(H = 5)

#x = np.array([np.arange(0,1,.001)])
x = np.array([np.arange(0,1,.0025)])
sigma = 0.1
eps = np.random.uniform(-0.1,0.1,x.shape) # uniform noise
t = x + 0.3*np.sin(2*np.pi*x) + eps # sin (x)

################################
#train on the forward problem
################################
#plt.scatter(x.T, t.T)
#tlp.train(x.T,t.T,0.003, 1000, batch = False)
#y = tlp.tlp(x.T)
#plt.plot(x.T, y, 'r-')

##show error
#plt.figure()
#plt.plot(tlp.E)


#################################
#train on the inverse problem
#################################

#plt.figure()
#plt.scatter(t.T, x.T)

#y = tlp.tlp(x.T)
#E0 = np.sum([tlp.E_n(y[i], x[0,i]) for i in range(x.shape[1])])
#tlp.train(t.T,x.T, 0.001, 100000, batch = False)
#print 'initial error was: ' + str(E0)
#y = tlp.tlp(x.T)
#plt.plot(x.T, y, 'r-')

#train a gaussian mixture density network
M = 3
mdn = MDN(H=5, d = 1, ny = 1, M = M)

# init weights to model the unconditional p(d)
mdn.init_weights(x.T, 100)

# now the network should approximate the distribution of the target data
plt.figure()
plt.hist(x[0],50, normed=True)
y = mdn.forward(x.T)

#make a plot
c = mdn.ny / (3*M)
alpha = y[:, 0:M]
sigma = y[:, M:2*M]
mu = y[:, 2*M:]
phi = np.zeros([M, x.shape[1]])
prior = np.zeros([x.shape[1]])
for k in range(M):
  phi[k, :] = (1.0 / (2*np.pi*sigma[:,k])**(0.5)) * np.exp(- 1.0 * (x[0,:]-mu[:,k])**2 / (2 * sigma[:,k]))
  prior = prior + alpha[:,k] * phi[k, :]
plt.figure()
plt.plot(x[0], prior)
plt.xlim([0,1])
plt.ylim([0,1])
plt.show()

mdn.train_BFGS(t.T,x.T, 0.5, 70)
#mdn.train_BFGS(x.T,t.T, 0.5, 70)

y = mdn.forward(x.T)
c = mdn.ny / (3*M)
alpha = y[:, 0:M]
sigma = y[:, M:2*M]
mu = y[:, 2*M:]

plt.figure()
plt.plot(x.T, alpha[:,0])
plt.plot(x.T, alpha[:,1])
plt.plot(x.T, alpha[:,2])
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('mixing coefficients')
plt.figure()
plt.plot(x.T, mu[:,0])
plt.plot(x.T, mu[:,1])
plt.plot(x.T, mu[:,2])
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('means')

# make a contour plot
k = 0
X = x[0,:]
T = t[0,:]
nx = X.shape[0]
nt = T.shape[0]
phi = np.zeros([M,nx, nt])
P_cond = np.zeros([nx, nt])

for xi in range(X.shape[0]):
  for ti in range(X.shape[0]):
    for k in range(M):
      phi[k,xi,ti] = (1.0 / (2*np.pi*sigma[xi,k])**(0.5)) * np.exp(- 1.0 * np.sum((X[ti]-mu[xi,k])**2) / (2 * sigma[xi,k]))
      P_cond[xi,ti] = P_cond[xi,ti] + phi[k,xi,ti] * alpha[xi,k]
f = plt.figure()
ax = Axes3D(f)
#ax.contour(X,Y,Z)
[X,Y]=np.meshgrid(X,X)
ax.plot_surface(X,Y,P_cond, rstride=10, cstride=10, cmap=cm.jet, linewidth=0, antialiased=False)
#plt.contour(X,Y,P_cond)

#p = np.zeros([x.shape[1], t.shape[1]])
#for xi in range(x.shape[1]):
  #for ti in range(t.shape[1]):
    #T = np.tile(t.T[xi], [M,1])
    #p[xi, ti] = np.sum(alpha[xi] * mdn._phi(T, mu[xi,:], sigma[xi,:]))

#show error
plt.figure()
plt.plot(mdn.E)

plt.show()
