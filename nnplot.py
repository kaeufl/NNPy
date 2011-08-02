# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import mlab

def plotPost2D(mdn, y, 
                rangex = [0, 1], rangey = [0, 1], 
                deltax = 0.01, deltay = 0.01,
                true_model = None):
  M = mdn.M
  alpha, sigma, mu = mdn.getMixtureParams(y)
  print 'mu: ' + str(mu)
  print 'sigma: ' + str(sigma)
  print 'true value: ' + str(true_model)
  xlin = np.arange(rangex[0], rangex[1], deltax)
  ylin = np.arange(rangey[0], rangey[1], deltay)
  [XLIN, YLIN] = np.meshgrid(xlin, ylin)
  
  phi = np.zeros([M,ylin.shape[0], xlin.shape[0]])
  P = np.zeros([ylin.shape[0], xlin.shape[0]])
  for k in range(M):
    phi[k,:,:] = mlab.bivariate_normal(XLIN, YLIN, np.sqrt(sigma[k]), np.sqrt(sigma[k]), mu[k,0], mu[k,1])
    P = P + phi[k,:,:] * alpha[k]
  plt.imshow(P, #interpolation='bilinear', 
            ##cmap=cm.gray,
            origin='lower', 
            extent=[rangex[0],rangex[1],
                    rangey[0],rangey[1]]
            )
  #plt.contour(XLIN, YLIN, P, 
              #levels = [0, 1.0/np.exp(1)]
  #            )
  #plt.scatter(true_model[0],true_model[1],marker='^', c="r")
  if not true_model == None:
    plt.axvline(true_model[0], c = 'r')
    plt.axhline(true_model[1], c = 'r')
  
def plotPost1D(mdn, y, rangex = [0, 1], deltax = 0.01, true_model = None):
  alpha, sigma, mu = mdn.getMixtureParams(y)
  xlin = np.arange(rangex[0], rangex[1], deltax)
  phi = np.zeros([mdn.M,xlin.shape[0]])
  P = np.zeros([xlin.shape[0]])
  for k in range(mdn.M):
    phi[k, :] = (1.0 / (2*np.pi*sigma[k])**(0.5)) * np.exp(- 1.0 * (xlin-mu[k,0])**2 / (2 * sigma[k]))
    P = P + phi[k, :] * alpha[k]
  #import pdb; pdb.set_trace()
  plt.plot(xlin, P)
  if true_model != None:
    plt.axvline(true_model, c = 'r')

def plotPostCond(mdn, x, t):
  y = mdn.forward(x)
  alpha, sigma, mu = mdn.getMixtureParams(y)
  N = t.shape[0]
  phi = np.zeros([mdn.M, N, N])
  P = np.zeros([N, N])
  T = np.tile(t, [N,1])
  for k in range(mdn.M):
    SIGMA = np.tile(sigma[k,:], [N, 1]).T
    MU = np.tile(mu[k,0,:], [N, 1]).T
    phi[k,:,:] = (1.0 / (2 * np.pi * SIGMA)**(0.5)) * np.exp(- 1.0 * (T-MU)**2 / (2 * SIGMA))
    P = P + phi[k,:,:] * np.tile(alpha[k, :], [N, 1]).T
  plt.imshow(P, #interpolation='bilinear', 
           #cmap=cm.gray,
           origin='lower', 
           extent=[min(t),max(t),
                   min(t),max(t)]
           )
  #X, Y = np.meshgrid(t, t)
  #plt.contour(X, Y, P, 
              #levels = [0, 1.0/np.exp(1)]
             #)

def plotPostMap(mdn, x, t):
  y = mdn.forward(x)
  alpha, sigma, mu = mdn.getMixtureParams(y)
  
  
  
             
def plotModelVsTrue(mdn, y, t, thres = 0.7, dim = 0):
  alpha, sigma, mu = mdn.getMixtureParams(y)
  # find most important kernels
  
  # find most probable kernel
  idx_max = np.argmax(alpha, axis = 0)
  mu_max = np.zeros([y.shape[0], mdn.c])
  sigma_max = np.zeros([y.shape[0]])
  alpha_max = np.zeros([y.shape[0]])
  for n in range(y.shape[0]):
    mu_max[n, :] = mu[idx_max[n],:,n]
    sigma_max[n] = sigma[idx_max[n], n]
    alpha_max = alpha[idx_max[n], n]
  
  if np.any(alpha_max <= thres):
    print 'Warning, not all mixture coefficients are above threshold.'
  
  plt.scatter(mu_max[:,dim], t)
  plt.xlim([min(t), max(t)])
  plt.ylim([min(t), max(t)])
  plt.xlabel('prediction')
  plt.ylabel('true value')
  
  
