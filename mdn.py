# -*- coding: utf-8 -*-
import numpy as np
from mlp import TLP
from datetime import datetime

class MDN(TLP):
  def __init__(self, H = 3, d = 1, ny = 1, M = 3, debug_output = False):
    """
    H: number of hidden units
    d: number of inputs
    ny: number of outputs
    M: number of mixture components
    """
    self.c = ny
    ny = 2*M + M * ny # M mixing coefficients + M variances + M*ny means
    self.M = M
    self.count_fwd = 0
    TLP.__init__(self, H, d, ny, linear_output = True, error_function = 'mdn', 
                   debug_output = debug_output)
    
  def init_weights(self, t, prior, scaled_prior = False):
    """
    initialize weights and biases so that the network models the unconditional density 
    of the target data p(t)
    t: target data
    prior: 1/sigma^2
    """
    from scipy.cluster.vq import kmeans2
    from scipy.spatial.distance import cdist
    
    # check if t has the right shape
    if not t.ndim == 2:
      t = t[:, None]
    
    if scaled_prior:
      #self.w1 = np.random.normal(loc=0.0, scale = 1,size=[H, d+1])/np.sqrt(d+1) # 1st layer weights + bias
      #self.w2 = np.random.normal(loc=0.0, scale = 1,size=[ny, H+1])/np.sqrt(H+1) # 2nd layer weights + bias
      sigma1 = 1.0/np.sqrt(self.d+1)
      sigma2 = 1.0/np.sqrt(self.H+1)
    else:
    # init weights from gaussian with width given by prior
      sigma1 = 1.0/np.sqrt(prior)
      sigma2 = sigma1
      
    self.w1 = np.random.normal(loc=0.0, scale = 1,size=[self.H, self.d+1]) * sigma1 # 1st layer weights + bias
    self.w2 = np.random.normal(loc=0.0, scale = 1,size=[self.ny, self.H+1]) * sigma2 # 2nd layer weights + bias
    
    # init biases (taken from netlab, gmminit.m)
    [centroid, label] = kmeans2(t, self.M)
    cluster_sizes = np.maximum(np.bincount(label), 1) # avoid empty clusters
    alpha = cluster_sizes/np.sum(cluster_sizes)
    if (self.M > 1):
      # estimate variance from the distance to the nearest centre
      sigma = cdist(centroid, centroid)
      sigma = np.min(sigma + np.diag(np.diag(np.ones(sigma.shape))) * 1000, 1)
      sigma = np.maximum(sigma, np.finfo(float).eps) # avoid underflow
    else:
      # only one centre: take average variance
      sigma = np.mean(np.diag([np.var(t)]))
    # set biases, taken from netlab, mdninit.m
    self.w2[0:self.M,0] = alpha
    self.w2[self.M:2*self.M,0] = np.log(sigma)
    self.w2[2*self.M:,0] = np.reshape(centroid, [self.M * self.c])
  
  def getMixtureParams(self, y):
    """Returns the parameters of the Gaussian mixture."""
    if len(y.shape) == 1:
      # avoid underrun
      alpha = np.maximum(y[0:self.M], np.finfo(float).eps)
      sigma = y[self.M:2*self.M]
      mu = np.reshape(y[2*self.M:], [self.c, self.M]).T
    else:
      # avoid underrun
      alpha = np.maximum(y.T[0:self.M], np.finfo(float).eps)
      sigma = y.T[self.M:2*self.M]
      mu = np.reshape(y[:, 2*self.M:], [y.shape[0], self.c, self.M]).T
    return alpha, sigma, mu
  
  def _phi(self, T, mu, sigma):
    # distance between target data and gaussian kernels    
    dist = np.sum((T-mu)**2, 1)
    phi = (1.0 / (2*np.pi*sigma)**(0.5*self.c)) * np.exp(- 1.0 * dist / (2 * sigma))
    # prevent underflow
    return np.maximum(phi, np.finfo(float).eps)
    
  def E_mdn(self, y, t, w1, w2):
    """mdn error function"""
    alpha, sigma, mu = self.getMixtureParams(y.T)
    #T = np.tile(t.T, [M,1,1])
    phi = self._phi(t.T[None, : :], mu, sigma)
    probs = np.maximum(np.sum(alpha * phi, 0), np.finfo(float).eps)
    return - np.log(probs)
  
  def dE_mdn(self, x, y, t, w1 = None, w2 = None):
    """derivative of mdn error function"""
    if w2 == None:
      w2 = self.w2
    M = int(self.M)
    # avoid underrun
    
    alpha, sigma, mu = self.getMixtureParams(y.T)
    #import pdb; pdb.set_trace()
    
    #T = t.T[None, None, :] # note: np.tile is slower than this notation
    T = t.T[None, :]
    
    phi = self._phi(T, mu, sigma)
    aphi = alpha*phi
    pi = aphi / np.sum(aphi, 0)
    
    # derivatives of E with respect to the output variables (s. Bishop 1995, chp. 6.4)
    dE_dy_alpha = alpha - pi
    dE_dy_sigma = - 0.5 * pi * ((np.sum((T-mu)**2 , 1) / sigma) - self.c)
    dE_dy_mu = pi[:,np.newaxis,:] * (mu - T) / sigma[:,np.newaxis,:]

    dk = np.zeros([self.ny, x.shape[0]])
    dk[0:M,:] = dE_dy_alpha
    dk[M:2*M,:] = dE_dy_sigma
    
    dk[2*M:] = np.reshape(dE_dy_mu, [M*self.c, x.shape[0]])
    
    # back-propagate the dks
    #t0=datetime.now()
    dEnw1, dEnw2 = self.backward(x, dk, w2)
    #print 'eval of dE_mdn:' + str((datetime.now()-t0))
    #dj = (1 - self.z[1:]**2) * np.dot(w2[:,1:].T, dk)
    # evaluate derivatives with respect to the weights
    #dEnw1 = (dj[:,:,np.newaxis]*x[np.newaxis,:,:]).transpose(1,0,2)
    #dEnw2 = (dk[:,:,np.newaxis]*self.z.T[np.newaxis,:,:]).transpose(1,0,2)
    return dEnw1, dEnw2
    
  def HE_mdn(self):
    pass
    
  def _tlp(self, x, w1 = None, w2 = None):
    # perform forward propagation    
    #t0=datetime.now()
    y = TLP._tlp(self, x, w1, w2)
    
    #self.count_fwd = self.count_fwd + 1
    
    # the outputs are ordered as follows:
    # y[0:M]:       M mixing coefficients alpha
    # y[M:2*M]:     M kernel widths sigma
    # y[2*M:M*c]:   M x c kernel centre components
    M = int(self.M)
    # calculate mixing coefficients, variances and means from network output
    y[0:M] = self.softmax(y[0:M]) # alpha
    # avoid overflow
    y[M:2*M] = np.minimum(y[M:2*M], np.log(np.finfo(float).max))
    y[M:2*M] = np.exp(y[M:2*M]) # sigma
    # avoid underflow
    y[M:2*M] = np.maximum(y[M:2*M], np.finfo(float).eps)
    #print (datetime.now()-t0).microseconds
    return y

