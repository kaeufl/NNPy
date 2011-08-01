# -*- coding: utf-8 -*-
import numpy as np

def trim(x, thres = np.finfo(float).eps):
  """remove leading and trailing zeros common to all rows"""
  for col in range(x.shape[1]):
    if np.any(np.abs(x[:,col]) >= thres):
      break
  x = x[:, col:]
  print col
  for col in range(x.shape[1]):
    if np.all(np.abs(x[:,col]) <= thres):
      break
  x = x[:,:col]
  print col
  return x
  
def downsample(x, N):
  """Return N samples from each row of x"""
  s = np.round(np.linspace(0, x.shape[1], N, False)).astype(int)
  return x[:, s]
  
def whiten(x, full = True):
  """
  Normalize input values to have zero mean and standard deviation one.
  """
  N = x.shape[0]
  mu = np.sum(x, axis = 0) / N
  if not full:
    sigma2 = 1.0/(N-1) * np.sum((x - mu)**2, axis = 0)
    #print sigma2
    #sigma2 = np.mean(sigma2)
    return (x-mu)/np.sqrt(sigma2)
  #else:
    #d = x - mu
    #SIGMA = 1.0/(N-1) * np.sum(d[:, None, :] * d[:, :, None], axis = 0)
    ##import pdb; pdb.set_trace()
    #l, U = np.linalg.eig(SIGMA)
    #L = np.diag(l)
    #return np.sum(np.dot(np.linalg.inv(np.sqrt(L)), U.T)[None, :, :] * d[:, :, None], axis = 1)
    
def rescale(x):
  """
  Rescale network input to fall into the range [-1;1] and to have zero mean
  """
  N = x.shape[0]
  mu = np.sum(x, axis = 0) / N
  x = x - mu
  return x / np.max(np.abs(x), axis = 0)[None, :]
  
def thres_filter(x, thres):
  """Set values below threshold to zero."""
  x[np.abs(x) <= thres] = 0.0
  return x