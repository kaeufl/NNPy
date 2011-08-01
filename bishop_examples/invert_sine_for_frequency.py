# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mlp import MDN
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

def plot_post(y, m):
  alpha = y[0]
  sigma = y[1]
  mu = y[2]
  print 'mu: ' + str(mu)
  print 'sigma: ' + str(mu)
  mlin = np.arange(-2, 2, 0.01)
  phi = (1.0 / (2*np.pi*sigma)**(0.5)) * np.exp(- 1.0 * (mlin-mu)**2 / (2 * sigma))
  plt.plot(mlin, phi)
  plt.axvline(m, c = 'r')
  
def forward(mdn, m, x):
  d = np.sin(2 * np.pi * np.outer(m,x))
  print 'm=' + str(m)
  #plt.figure()
  #plt.plot(x, d.T)
  #plt.ylim([-1,1])
  y = mdn.forward(d)
  plot_post(y[0], m)

#########################
# create training dataset
# sine
np.random.seed(42)

omegamin = 0.5
omegamax = 2
omegalin = np.arange(omegamin, omegamax, 0.1)
omega = np.random.uniform(omegamin, omegamax, size = 200)
a = np.random.uniform(0.1, 1, size = 200)
#a=np.ones(omega.shape[0])
m = np.array([a, omega]).T
x = np.arange(-0.5,0.5,.0125)
d = np.zeros([m.shape[0], x.shape[0]])
for mi in range(m.shape[0]):
  d[mi,:] = m[mi,0] * np.sin(m[mi,1] * 2 * np.pi * x)

plt.figure()
plt.plot(x, d.T)

#############################
# forward problem
#mdn = MDN(H=20, d = 1, ny = 10, M = 1)
#mdn.init_weights(m, 10)


#############################
# inverse problem
mdn = MDN(H=5, d = 80, ny = 1, M = 1)
np.random.seed(42)
mdn.init_weights(m[:,1], 1e3)
mdn.train_BFGS(d, m[:,1], 1e-5, 100)

#y = mdn.forward(d)


############################
# generalize!
#m = np.random.uniform(mmin, mmax, size=1)
m_test = np.arange(omegamin, omegamax, 0.01)
d = np.zeros([m_test.shape[0], x.shape[0]])
for mi in range(m_test.shape[0]):
  d[mi,:] = np.sin(m_test[mi] * 2 * np.pi * x)
y = mdn.forward(d)

plt.figure()
plt.stem(m_test, y[:,2]-m_test)

plt.figure()
plot_post(y[100], m_test[100])