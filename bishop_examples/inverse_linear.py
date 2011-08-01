# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mlp import MDN, plotPost1D
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

#def plot_post(y, m):
  #alpha = y[0]
  #sigma = y[1]
  #mu = y[2]
  #print 'mu: ' + str(mu)
  #print 'sigma: ' + str(mu)
  #mlin = np.arange(-2, 2, 0.005)
  #phi = (1.0 / (2*np.pi*sigma)**(0.5)) * np.exp(- 1.0 * (mlin-mu)**2 / (2 * sigma))
  #plt.plot(mlin, phi)
  #plt.axvline(m, c = 'r')
  
def forward(mdn, m, x):
  d = np.outer(m,x)
  #print 'm=' + str(m)
  #plt.figure()
  #plt.plot(x, d.T)
  #plt.ylim([-1,1])
  y = mdn.forward(d)
  return y
  #plot_post(y[0], m)
#
#########################
# create training dataset 
mmin = 0
mmax = 2
#mmin = 0.3
#mmax = 0.7
m = np.array([np.random.uniform(mmin, mmax, size = 300)]).T
#m = np.random.normal(loc=0, scale = 1, size = 100)
#x = np.arange(-1,1,.1)
x = np.array([-.5, .5])
#x = np.array([1, 1.5])
d = np.outer(m,x)

# add gaussian noise
#eps = np.random.normal(loc = 0.0, scale = 0.05, size = d.shape) # uniform noise
#d = d + eps

plt.figure()

plt.subplot(221)
plt.plot(x, d.T)
plt.title("Training set")


#############################
# forward problem
#mdn = MDN(H=20, d = 1, ny = 10, M = 1)
#mdn.init_weights(m, 10)


#############################
# inverse problem
mdn = MDN(H=1, d=2, ny=1, M=1)

# prior value large -> initial weights near zero -> prior network output independent of input, but very poor learning performance
mdn.init_weights(m, prior = 1e3)
print 'network weights after initialisation'
print mdn.w1
print mdn.w2

# plot prior
plt.subplot(223)
plt.hold(True)
for k in np.arange(mmin, mmax, 0.01):
  y = mdn.forward(k * x)
  plotPost1D(mdn, y, [mmin, mmax], 0.01)
plt.title('prior network output (test set)')

# train
mdn.train_BFGS(d, m, 1e-5, 100)

#validate training set
plt.subplot(222)
y = mdn.forward(d)
plt.scatter(y[:,2], m)
plt.title('slopes (training set)')
plt.xlabel('prediction')
plt.ylabel('true value')

############################
# generalize!
#m = np.random.uniform(mmin, mmax, size=1)

plt.subplot(224)
for k in np.arange(mmin, mmax, 0.1):
  y = forward(mdn, k, x)
  #import pdb; pdb.set_trace()
  plotPost1D(mdn, y, [mmin, mmax], 0.01, k)

plt.title('posterior network output (test set)')

#plt.suptitle("Inverting for the slope of lines with random uniform noise added. (1 hidden unit, 1 mixture component)")
plt.suptitle("Inverting for the slope of lines. (noise-free, 1 hidden unit, 1 mixture component)")

plt.show()

