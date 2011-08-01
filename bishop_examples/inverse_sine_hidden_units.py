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
marginalize = True
train_with_noise = False
test_with_noise = False

noise_width = 0.05

#omegamin = 0.1
#omegamax = 0.2
#amin = 0.1
#amax = 0.2

omegamin = 0.1
omegamax = 1
amin = 0.1
amax = 1

N = 400 # size of training set
H = 10 # no of hidden units


###############################################################################
# notes
# as expected: hidden units which have comparably small 1st layer
# weights have little impact on the output
###############################################################################

#########################
# create training dataset
# sine
np.random.seed(10)
omegalin = np.arange(omegamin, omegamax, 0.1)
omega = np.random.uniform(omegamin, omegamax, size = N)
a=np.random.uniform(amin, amax, size = N)
#a=np.ones(omega.shape[0])
m = np.array([a, omega]).T
#x = np.arange(-0.5,0.5,.025) # that was used for the somewhat working version
#x = np.arange(0,1,.025)
x = np.arange(0,1,1.0/40)
d = np.zeros([m.shape[0], x.shape[0]])
for mi in range(m.shape[0]):
  d[mi,:] = m[mi,0] * np.sin(m[mi,1] * 2 * np.pi * x)
  
# add gaussian noise
if train_with_noise:
  eps = np.random.normal(loc = 0.0, scale = noise_width, size = d.shape)
  d = d + eps
  
#############################
# train on marginal pdf of frequency
mdn = MDN(H=H, d = 40, ny = 1, M = 1)
mdn.init_weights(np.array([m[:,1]]).T, 1e1)
mdn.train_BFGS(d, m[:,1], 1e-5, 100) # bfgs

plt.figure()

y = mdn.forward(d)
plt.subplot(211)
plt.plot(mdn.w1.T)
plt.title('1st layer weights')
plt.xlabel('input node')
mu = y[:,2]
plt.subplot(212)
plt.scatter(mu, m[:,1], c = m[:,0])
plt.title('10 hidden units')
plt.xlabel('prediction')
plt.ylabel('true value')

w1_bak = mdn.w1.copy()
w2_bak = mdn.w2.copy()

plt.figure();

def switch_off(unit):
  if unit < 5:
    pos2 = unit
    pos1 = 5 + unit
  else:
    pos2 = 5 + unit
    pos1 = 10 + unit
  mdn.w1[unit] = 0
  y = mdn.forward(d)
  plt.subplot(4,5,pos1+1)
  plt.plot(mdn.w1.T)
  plt.title('1st layer weights')
  plt.xlabel('input node')
  mu = y[:,2]
  plt.subplot(4,5,pos2+1)
  plt.scatter(mu, m[:,1], c = m[:,0])
  plt.title('Unit '+ str(unit+1) +' switched off')
  plt.xlabel('frequency')
  plt.ylabel('true value (color: amplitude)')
  mdn.w1 = w1_bak.copy()
  mdn.w2 = w2_bak.copy()

for u in range(10):
  switch_off(u)

plt.suptitle('Switching off hidden units', size=14)
plt.show()