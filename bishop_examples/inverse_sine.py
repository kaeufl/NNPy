# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mdn import MDN, plotPostCond, plotPost1D
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


###working version###
#omegamin = 0.5
#omegamax = 2
#amin = 0.5
#amax = 1 # does work well
#amax = 20 # does not work well
######

###notes:
# network tends to overestimate samples with low frequencies
# while at the same time underestimating their amplitude
# if the range of x is increased such that at least half of a period
# is included, then it gets better
#
# if frequencysettings are made properly but amplitudes are varied from 0.1 to 1 then
# the frequency of samples with amplitude < .4 are overestimated for small frequencies
# and underestimated for large frequencies
# if amplitude is allowed to vary only on a very small interval e.g. 0.1 to 0.15
# then it works again
#
# two hidden units seem to be enough,in the case of more units the first layer weights
# roughly form two groups after training, which oscillate and are phase shifted
#
# it might be useful to use more than one mixture component even if the problem is
# unique, since the Gaussian kernels are spherical, with a single width parameters
#
# addressing the bias-problem for small and high frequency-values:
#  - increasing the number of hidden units seems to help: H = 2 strong effect, 
#    H=10 much better
#    the network is then able to learn the training set correctly
# - it does again not work, even for a large number of hidden units if the 
#   training set is too small (N=400 works)


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

plt.figure()

plt.subplot(231)
plt.plot(x, d.T)
plt.title('400 training patterns')

###############################
# train on the inverse problem
if marginalize:
  H = 10 # no of hidden units
  
  #############################
  # train on marginal pdf of frequency
  mdn = MDN(H=H, d = 40, ny = 1, M = 1)
  mdn.init_weights(np.array([m[:,1]]).T, 1e1)
  mdn.train_BFGS(d, m[:,1], 1e-5, 100) # bfgs
else:
  H = 10 # no of hidden units
  
  ############################
  # train on joint pdfs of amplitude and frequency
  mdn = MDN(H=H, d = 40, ny = 2, M = 1)
  mdn.init_weights(m, 1e5)
  #y = mdn.forward(d)
  #plt.figure()
  #mlp.plotPost2D(mdn, y[0], [amin,amax], [omegamin,omegamax],
  #           0.01,0.01)
  #plt.show()
  mdn.train_BFGS(d, m, 1e-5, 200, constrained = False) # bfgs
  #mdn.train_BFGS(d, m, 1e-7, 200, constrained = True) # l_bfgs_b
  

############################
# evaluate the training set
if marginalize:
  y = mdn.forward(d)
  mu = y[:,2]
  plt.subplot(232)
  plt.scatter(mu, omega)
  plt.title('frequencies (training set)')
  plt.xlabel('prediction')
  plt.ylabel('true value')
else:
  y = mdn.forward(d)
  mu_a = y[:,2]
  mu_omega = y[:,3]
  #plt.figure()
  plt.subplot(232)
  plt.scatter(mu_a, a, c = mu_omega)
  plt.colorbar()
  plt.xlim([amin, amax])
  plt.ylim([amin, amax])
  plt.title('amplitudes (training set)')
  plt.xlabel('prediction')
  plt.ylabel('true value')
  
  #plt.figure()
  plt.subplot(233)
  plt.scatter(mu_omega, omega, c = mu_a)
  plt.colorbar()
  plt.title('frequencies (training set)')
  plt.xlim([omegamin,omegamax])
  plt.ylim([omegamin,omegamax])
  plt.xlabel('prediction')
  plt.ylabel('true value')

############################
# generalize!
#omega_test = np.arange(omegamin, omegamax, 0.01)
#a_test = 0.5*np.ones(omega_test.shape[0])

N_test = 1000.0

omegalin_test = np.arange(omegamin, omegamax, (omegamax-omegamin)/N_test)
omega_test = np.random.uniform(omegamin, omegamax, size = N_test)
#omega_test = omegalin_test
a_test = np.random.uniform(amin, amax, size = N_test)
#a_test = np.ones(omega_test.shape[0])
m_test = np.array([a_test, omega_test]).T

d_test = np.zeros([m_test.shape[0], x.shape[0]])
for mi in range(m_test.shape[0]):
  d_test[mi,:] = a_test[mi] * np.sin(omega_test[mi] * 2 * np.pi * x)

if test_with_noise:
  eps = np.random.normal(loc = 0.0, scale = noise_width, size = d_test.shape)
  d_test = d_test + eps

y_test = mdn.forward(d_test)
sigma = y_test[:,1]
mu = y_test[:,2:]

if marginalize: 
  plt.subplot(235)
  plt.scatter(y_test[:,2], omega_test)
  plt.title('frequencies (test set)')
  plt.xlabel('prediction')
  plt.ylabel('true value')
  
  plt.subplot(236)
  #omegalin_test = np.arange(omegamin, omegamax, (omegamax-omegamin)/10000.0)
  mlin_test = np.array([a_test[0:N_test], omegalin_test]).T
  dlin_test = np.zeros([mlin_test.shape[0], x.shape[0]])
  for mi in range(mlin_test.shape[0]):
    dlin_test[mi,:] = a_test[mi] * np.sin(omegalin_test[mi] * 2 * np.pi * x)
  if test_with_noise:
    eps = np.random.normal(loc = 0.0, scale = noise_width, size = dlin_test.shape)
    dlin_test = dlin_test + eps
  plotPostCond(mdn, dlin_test, omegalin_test)
  plt.title('$p(freq|freq_{true})$')
  plt.xlabel('prediction')
  plt.ylabel('true value')
  
  plt.subplot(234)
  plotPost1D(mdn, y_test[500], [omegamin,omegamax],
                 0.01, m_test[500, 1])
  plt.title('marginal pdf for a pattern from the test set')
else:
  plt.subplot(235)
  plt.scatter(y_test[:,2], a_test, c = y_test[:,3])
  plt.colorbar()
  plt.xlim([amin,amax])
  plt.ylim([amin, amax])
  plt.title('amplitudes (test set, 10000 patterns)')
  plt.xlabel('prediction')
  plt.ylabel('true value')
  
  plt.subplot(236)
  plt.scatter(y_test[:,3], omega_test, c = y_test[:,2])
  plt.colorbar()
  plt.xlim([omegamin,omegamax])
  plt.ylim([omegamin,omegamax])
  plt.title('frequencies (test set, 10000 patterns)')
  plt.xlabel('prediction')
  plt.ylabel('true value')

  plt.subplot(234)
  plotPost2D(mdn, y_test[500], [amin,amax], [omegamin,omegamax],
                 0.01,0.01, m_test[500])
  plt.xlabel('amplitude')
  plt.ylabel('frequency')
  plt.title('joint pdf for a pattern from the test set')
  
#plt.figure()
#plt.stem(omega_test, y[:,3]-omega_test)

#plt.figure()
#plt.stem(range(a_test.shape[0]), y[:,2]-a_test)

#plt.suptitle("Noisy training set, noisy test set.")
plt.suptitle('omegamin=%s, omegamax=%s, amin=%s, amax=%s, marginalize=%s, train_with_noise=%s, test_with_noise=%s, H=%s' %\
            (omegamin, omegamax, amin, amax, marginalize, train_with_noise, test_with_noise, H))
plt.show()