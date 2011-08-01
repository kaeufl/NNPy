# -*- coding: utf-8 -*-
################################
# Notes:
# - N = 1000, omegamin = 0.1, omegamax = 1, amin = 0.1, amax = 1, H = 10
# lower frequencies (less than half of a period sampled) become pretty noisy
# remainder works (forward_sine_small_range.pdf)
#
# - N = 1000, omegamin = 0.1, omegamax = 2, amin = 0.1, amax = 1, H = 10
# fails (forward_sine_large_range_fail.pdf)
#
#- N = 2000, omegamin = 0.1, omegamax = 2, amin = 0.1, amax = 1, H = 10
# fails as well (forward_sine_large_range_fail2.pdf)
#
# - N = 2000, omegamin = 0.1, omegamax = 2, amin = 0.1, amax = 1, H = 20
# better but still not good (forward_sine_large_range_fail3.pdf)
# sines showing 1/2 up to 2 periods seem to work best, lower frequencies worst
#
#
################################

import numpy as np
import matplotlib.pyplot as plt
import mlp
from mlp import MDN
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm, mlab

np.seterr(invalid='raise')  

#########################
# parameters
train_with_noise = False
test_with_noise = False

noise_width = 0.001

omegamin = 0.1
omegamax = 2
amin = 0.1
amax = 1

N = 2000 # size of training set

#########################
# create training dataset
# sine
np.random.seed(11)
omegalin = np.arange(omegamin, omegamax, 0.1)
omega = np.random.uniform(omegamin, omegamax, size = N)
a=np.random.uniform(amin, amax, size = N)
#a=np.ones(omega.shape[0])
m = np.array([a, omega]).T
x = np.arange(0,1,1.0/40)
d = np.zeros([m.shape[0], x.shape[0]])
for mi in range(m.shape[0]):
  d[mi,:] = m[mi,0] * np.sin(m[mi,1] * 2 * np.pi * x)
  
# add gaussian noise
if train_with_noise:
  eps = np.random.normal(loc = 0.0, scale = noise_width, size = d.shape)
  d = d + eps
  
plt.figure()
plt.subplot(121)
plt.plot(x, d.T)
plt.title(str(N) + ' training patterns')

def train_marg(mdn, m, d):
  mdn.init_weights(d, 1e1)
  mdn.train_BFGS(m, d,1e-5, 200) # bfgs
  return mdn

nets = list()
for di in range(1, d.shape[1]):
  mdn = MDN(H=20, d = 2, ny = 1, M = 1)
  mdn = train_marg(mdn, m, np.array([d[:,di]]).T)
  nets.append(mdn)
  
# test
N_test = 20.0
omega_test = np.random.uniform(omegamin, omegamax, size = N_test)
a_test     = np.random.uniform(amin, amax, size = N_test)

plt.subplot(122)
for ti in range(N_test):
  y = np.zeros([d.shape[1], 3])
  for di in range(1, d.shape[1]):
    y[di, :] = nets[di-1].forward(np.array([a_test[ti], omega_test[ti]]))
  plt.plot(x, y[:, 2])
plt.title('Means of posterior marginals vs. output node (test set)')

plt.suptitle('Forward problem, 40 networks trained on marginal distributions of output components')
plt.show()