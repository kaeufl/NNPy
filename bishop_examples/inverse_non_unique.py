# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import mlp
from mlp import MDN
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm, mlab

np.seterr(invalid='raise')  
  
#########################
# parameters
marginalize = False
train_with_noise = False
test_with_noise = False

omegamin = 0.5
omegamax = 2
amin = 0.5
amax = 1 # does work well
#amax = 20 # does not learn the training set properly

#########################
# create training dataset
# sine
np.random.seed(42)
omegalin = np.arange(omegamin, omegamax, 0.1)
omega = np.random.uniform(omegamin, omegamax, size = 400)
a=np.random.uniform(amin, amax, size = 400)
#a=np.ones(omega.shape[0])
m = np.array([a, omega]).T
x = np.arange(-0.5,0.5,.025)
d = np.zeros([m.shape[0], x.shape[0]])
for mi in range(m.shape[0]):
  d[mi,:] = m[mi,0] * np.sin(m[mi,1] * 2 * np.pi * x)
  
# add gaussian noise
if train_with_noise:
  eps = np.random.normal(loc = 0.0, scale = 0.05, size = d.shape)
  d = d + eps

plt.figure()

plt.subplot(231)
plt.plot(x, d.T)
plt.title('400 training patterns')

###############################
# train on the onverse problem
if marginalize:
  #############################
  # train on marginal pdf of frequency
  mdn = MDN(H=5, d = 40, ny = 1, M = 1)
  #np.random.seed(41)
  mdn.init_weights(m[:,1], 1e3)
  mdn.train_BFGS(d, m[:,1], 1e-5, 100) # bfgs
else:
  ############################
  # train on joint pdfs of amplitude and frequency
  mdn = MDN(H=5, d = 40, ny = 2, M = 1)
  #np.random.seed(41)
  mdn.init_weights(m, 1e2)
  mdn.train_BFGS(d, m, 1e-5, 200) # bfgs
  #mdn.train_BFGS(d, m, 1e-5, 1000) # l_bfgs_b
  

############################
# evaluate the training set
if marginalize:
  y = mdn.forward(d)
  mu = y[:,2]
  plt.figure()
  plt.scatter(mu, omega)
  plt.title('frequencies (training set)')
  plt.xlabel('prediction')
  plt.ylabel('true value')
else:
  y = mdn.forward(d)
  mu = y[:,2]
  #plt.figure()
  plt.subplot(232)
  plt.scatter(mu, a)
  plt.title('amplitudes (training set)')
  plt.xlabel('prediction')
  plt.ylabel('true value')
  mu = y[:,3]
  #plt.figure()
  plt.subplot(233)
  plt.scatter(mu, omega)
  plt.title('frequencies (training set)')
  plt.xlabel('prediction')
  plt.ylabel('true value')

############################
# generalize!
#omega_test = np.arange(omegamin, omegamax, 0.01)
#a_test = 0.5*np.ones(omega_test.shape[0])

omega_test = np.random.uniform(omegamin, omegamax, size = 10000)
a_test = np.random.uniform(amin, amax, size = 10000)
#a_test = np.ones(omega_test.shape[0])
m_test = np.array([a_test, omega_test]).T

d_test = np.zeros([m_test.shape[0], x.shape[0]])
for mi in range(m_test.shape[0]):
  d_test[mi,:] = a_test[mi] * np.sin(omega_test[mi] * 2 * np.pi * x)

if test_with_noise:
  eps = np.random.normal(loc = 0.0, scale = 0.05, size = d_test.shape)
  d_test = d_test + eps

y_test = mdn.forward(d_test)
sigma = y_test[:,1]
mu = y_test[:,2:]

if marginalize:
  plt.figure()
  plt.scatter(y_test[:,2], omega_test)
  plt.title('frequencies (test set)')
  plt.xlabel('prediction')
  plt.ylabel('true value')
else:
  #plt.figure()
  plt.subplot(236)
  plt.scatter(y_test[:,3], omega_test)
  plt.title('frequencies (test set, 10000 patterns)')
  plt.xlabel('prediction')
  plt.ylabel('true value')

  #plt.figure()
  plt.subplot(235)
  plt.scatter(y_test[:,2], a_test)
  plt.title('amplitudes (test set, 10000 patterns)')
  plt.xlabel('prediction')
  plt.ylabel('true value')
  
  #plt.figure()
  plt.subplot(234)
  mlp.plotPost2D(mdn, y_test[1000], [amin,amax], [omegamin,omegamax],
                 0.01,0.01, m_test[1000])
  plt.xlabel('amplitude')
  plt.ylabel('frequency')
  plt.title('joint pdf for a pattern from the test set')
  
#plt.figure()
#plt.stem(omega_test, y[:,3]-omega_test)

#plt.figure()
#plt.stem(range(a_test.shape[0]), y[:,2]-a_test)

plt.suptitle("Noisy training set, noisy test set.")
plt.show()