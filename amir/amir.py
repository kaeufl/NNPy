# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mdn import MDN
import nndata
import nnplot

N = 1000
# stations
st = np.array([[3.0, 15.0, 0.0], 
               [3.0, 16.0, 0.0],
               [4.0, 15.0, 0.0],
               [4.0, 16.0, 0.0],
               [5.0, 15.0, 0.0],
               [5.0, 16.0, 0.0]])
               
# observed traveltime data
d_obs = np.array([3.2802441,
                3.4058774,
                3.1368775,
                3.2680271,
                3.0000000,
                3.1368775])

def calc_T(x, m):
  n_obs = x.shape[0]
  d = np.zeros([n_obs])
  for xi in range(n_obs):
    d[xi] = 1.0/m[3] * np.sqrt(
      (x[xi, 0] - m[0])**2 + 
      (x[xi, 1] - m[1])**2 + 
      (x[xi, 2] - m[2])**2)
  return d
  
# generate training and test set
m_test = np.zeros([N, 4])
m_test[:, 0] = np.random.uniform(-20, 20, size = N)
m_test[:, 1] = np.random.uniform(-20, 20, size = N)
m_test[:, 2] = 5.0 # np.random.uniform(0, 15, size = N)
m_test[:, 3] = 5.0 # np.random.uniform(1, 10, size = N)

d_test = np.zeros([N, st.shape[0]])
for n in range(N):
  d_test[n] = calc_T(st, m_test[n])

# add noise
#eps = np.random.normal(loc = 0.0, scale = 0.1, size = d_test.shape)
#d_test = d_test + eps

# preprocess
d_test = nndata.whiten(d_test, False)
#d_test = nndata.rescale(d_test)

m_train = m_test[0:.7*N]
m_test = m_test[.7*N:]
d_train = d_test[0:.7*N]
d_test = d_test[.7*N:]

# train on X and Y
mdn = MDN(H = 10, d = 6, ny = 2, M = 3)
mdn.init_weights(m_train[:, 0:2], 1e3, scaled_prior = True)
mdn.train_BFGS(d_train, m_train[:,0:2], 1e-5, 500, constrained = False)

plt.subplot(2,2,1)
y = mdn.forward(d_test)
nnplot.plotModelVsTrue(mdn, y, m_test[:,0], dim = 0)
plt.title('X')

plt.subplot(2,2,2)
nnplot.plotModelVsTrue(mdn, y, m_test[:,1], dim = 1)
plt.title('Y')

# invert observations
plt.subplot(2,2,3)
d_obs = nndata.whiten(d_obs, False)
y = mdn.forward(d_obs) 

nnplot.plotPost2D(mdn, y[0,:], 
                  rangex = [-20, 20], rangey = [-20, 20], 
                  deltax = 0.01, deltay = 0.01,
                  true_model = None)
plt.title('Posterior probability density for X,Y')
# plot station locations
plt.scatter(st[:, 0], st[:, 1])

# plot true model
# x=15km, y=5km, z=5km
plt.scatter(15, 5, color='red')

plt.show()
