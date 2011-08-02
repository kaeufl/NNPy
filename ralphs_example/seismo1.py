# -*- coding: utf-8 -*-
import numpy as np
from mdn import MDN
from nnplot import plotModelVsTrue, plotPostMap
import nndata
import matplotlib.pyplot as plt
import os

np.seterr(invalid='raise')
#np.random.seed(42)

#############################################
# Notes:
# 
# working configurations:
# -noise free data set, H=6, M=1, nd = 50
# 
#
# scg training seems to get more stable if using a higher number of hidden units
#
#
# ----------------------------
# pre rpocessing issues
#   - scg, N=699, nd=50, H=10, M=1, nit=600:
#     works somewhat if input is whitened, rescaling whole seismograms does not 
#     work (does not converge)
#     
#     works similarly well, if every input is normalized to lie within [-1,1] and the 
#     means are corrected to 0
# 
#     

N_max = 999
N = 699
nd = 50

H = 10
M = 1
nit = 300

params = {0:'$\\rho$ lower',
          1:'$v_p$ lower',
          2:'$v_s$ lower',
          3:'$\\rho$ upper',
          4:'$v_p$ upper'}

noise_width = 1.0e-5

# read in ralph's training set
# use N patterns as test set, N_max-N as training set
d = np.zeros([N, 1600])
t = np.zeros([N, 6])
d_test = np.zeros([N_max-N, 1600])
t_test = np.zeros([N_max-N, 6])
for n in range(N):
  d[n] = np.fromfile(os.getcwd() + '/training/setup_1/model_' + str(n+1) + '/Uz_file_double.bin')
  t[n] = np.loadtxt(os.getcwd() + '/training/setup_1/model_' + str(n+1) + '/model_par.txt')
for n in range(N_max - N):
  d_test[n] = np.fromfile(os.getcwd() + '/training/setup_1/model_' + str(N+n+1) + '/Uz_file_double.bin')
  t_test[n] = np.loadtxt(os.getcwd() + '/training/setup_1/model_' + str(N+n+1) + '/model_par.txt')


##############################################
# PREPROCESSING
plt.figure()
plt.subplot(3,2,1)
plt.plot(d.T)
plt.title("initial training set")

# filter values below noise level
#d = nndata.thres_filter(d, noise_width * 100)
#d_test = nndata.thres_filter(d_test, noise_width * 100)
#plt.subplot(3,2,2)
#plt.plot(d.T)
#plt.title("threshold filter")

# trim
d = nndata.trim(d)
d_test = nndata.trim(d_test)
plt.subplot(3,2,3)
plt.plot(d.T)
plt.title("remove leading and tailing zeros")

# add noise
eps = np.random.normal(loc = 0.0, scale = noise_width, size = d.shape)
d = d + eps
eps = np.random.normal(loc = 0.0, scale = noise_width, size = d_test.shape)
d_test = d_test + eps

# down-sample
d = nndata.downsample(d, nd)
d_test = nndata.downsample(d_test, nd)
plt.subplot(3,2,4)
plt.plot(d.T)
plt.title("downsample")

#whiten
d = nndata.whiten(d, False)
d_test = nndata.whiten(d_test, False)
plt.subplot(3,2,5)
plt.plot(d.T)
plt.title("whiten")

#rescale
#d = nndata.rescale(d)
#d_test = nndata.rescale(d_test)
#plt.subplot(3,2,6)
#plt.plot(d.T)
#plt.title("rescale")

plt.suptitle("Data preprocessing.")
plt.show()

##############################################
# TRAINING
plt.figure()
plt.subplot(4,2,1)
plt.plot(d.T)
plt.title('training set')

plt.subplot(4,2,2)
plt.hist(d[:, 25], 20)
plt.title('initial distribution for input node 25')

nets = []

# train
for ti in range(t.shape[1]-1):
  mdn = MDN(H=H, d = nd, ny = 1, M = M)
  mdn.init_weights(t[:,ti], 1e3, scaled_prior = True)
  mdn.train_BFGS(d, t[:,ti], 1e-5, nit, constrained = False)
  nets.append(mdn)
  y = nets[ti].forward(d)
  plt.subplot(4,2,ti+3)
  plotModelVsTrue(mdn, y, t[:,ti])
  plt.title(params[ti])

plt.suptitle('setup_1, (%d patterns), %d hidden units, %d mixture component, %d input nodes, %d iterations' %\
             (N, H, M, nd, nit))

# test
plt.figure()
plt.subplot(4,2,1)
plt.plot(d_test.T)
plt.title('test set')
for ti in range(t_test.shape[1]-1):
  y_test = nets[ti].forward(d_test)
  plt.subplot(4,2,ti+3)
  plotModelVsTrue(mdn, y_test, t_test[:,ti])
  plt.title(params[ti])

plt.suptitle('setup_1, (%d patterns), %d hidden units, %d mixture component, %d input nodes, %d iterations' %\
             (N_max-N, H, M, nd, nit))
plt.show()