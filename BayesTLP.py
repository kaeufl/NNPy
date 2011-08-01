# -*- coding: utf-8 -*-
from mlp import TLP
import numpy as np

class BayesTLP(TLP):
  def __init__(self, H = 3, d = 1, ny = 1):
    TLP.__init__(self, H, d, ny, linear_output = True, error_function = 'bayes')
    
  def E_prior(self, w1, w2):
    return 0.5 * np.sum(np.append(w1.flatten(), w2.flatten())**2)
    
  def E_data(self, y, t):
    return self.E_sum_of_squares(y, t)
    #return 0.5 * np.sum((y - t.T)**2, 0)
    
  def E_bayes(self, y, t, w1, w2):
    """Regularized error function corresponding to Gaussian noise and a 
    Gaussian prior with zero mean"""
    return self.beta * self.E_data(y, t) + self.alpha * self.E_prior(w1, w2)
    
  def dE_bayes(self, x, y, t, w1, w2):
    """Derivatives of the regularized error function."""
    #import pdb; pdb.set_trace()
    dEn1_data, dEn2_data = self.dE_sum_of_squares(x, y, t, w1, w2)
    dEn1_prior, dEn2_prior = (w1, w2)
    # don't add prior weight contribution to the biases
    #dEn1_prior[:,0] = 0
    #dEn2_prior[:,0] = 0
    return (self.beta * dEn1_data + self.alpha * dEn1_prior, 
            self.beta * dEn2_data + self.alpha * dEn2_prior)
            
  def HE_bayes(self, x, y, t, w1, w2):
    """Hessian of the regularized error function"""
    H = self.HE_sum_of_squares(x, y, t, w1, w2)
    A = self.beta * H + self.alpha * np.eye(w1.size + w2.size)
    return A
  
  #def H_bayes(self, x, y, t):
    #"""Exact evaluation of the Hessian as described in Bishop 1995, 4.10"""
    #nw1 = self.w1.shape[0] * self.w1.shape[1]
    #nw2 = self.w2.shape[0] * self.w2.shape[1]
    ## the Hessian for the second layer weights is stored in a five-dimensional 
    ## array ordered as follows: [n, k, j, k', j']
    ## n = 0..N-1, k = 0..self.ny, j = 0..self.H
    
    #dk = (y - t) * self.beta
    #x_i = x[:, None, :, None, None]
    #x_i_prime = x[:, None, None, None, :]
    #z_j = self.z.T[:, None, :, None, None]
    #z_j_prime = self.z.T[:, None, None, None, :]
    #H_k = self.beta
    ## first derivative of activation function g'
    #g_prime_a_j_prime = 1 - self.z.T[:, None, None, :, None]**2
    #g_prime_a_j = 1 - self.z.T[:, :, None, None, None]**2
    ## second deriavative of activation function at a_j'
    #g_dprime_a_j_prime = 2 * (self.z.T[:, None, None, :, None]**2 - self.z.T[:, None, None, :, None])    
    
    #H2 = z_j * z_j_prime * np.diag(np.ones(self.w2.shape[0]))[None, :, None, :, None] * H_k
    ## Hessian with respect to first layer weights: [n, j, i, j', i']
    #H1 = x_i * x_i_prime \
         #* g_dprime_a_j_prime \
         #* np.diag(np.ones(self.w2.shape[1]))[None, :, None, :, None] \
         #* np.sum(self.w2[None, :, :] * dk[:,:,None], axis = 1)[:, None, None, :, None] \
         #+ x_i * x_i_prime \
         #* g_prime_a_j_prime \
         #* g_prime_a_j \
         #* np.sum(self.w2[:, :, None] * self.w2[:, None, :] * self.beta, axis = 0)[None, :, None, :, None]
    ## one weight in each of the layers
    #H12 = x_i * g_prime_a_j \
          #* (dk[:, None, None, :, None] * np.diag(np.ones(self.w2.shape[1]))[None, :, None, None, :] \
            #+ z_j_prime * self.w2.T[None, :, None, :, None] * H_k)
    
  def re_estimate(self, x, t):
    y = self.forward(x)
    X = np.append(np.ones([x.shape[0],1]),x,1)
    H = self.HE_sum_of_squares(X, y, t.T, self.w1, self.w2)
    
    (evl, evc) = np.linalg.eig(H)
    # note: there may be negative eigenvalues of H since the Hessian is evaluated at the minimum
    # of the regularized error function, furthermore the optimal point in weights space may not 
    # yet be reached
    # taken from netlab: just set the negative eigenvalues to zero
    print 'alpha_old = ' + str(self.alpha)
    print 'beta_old = ' + str(self.beta)
    
    print evl
    evl[evl<0] = np.finfo(float).eps
    evl = evl * self.beta
    
    E_W = self.E_prior(self.w1, self.w2)
    
    E_D = self.E_data(y, t.T)
    
    print 'E_W = ' + str(E_W)
    print 'E_D = ' + str(E_D)
    
    gamma = np.sum(evl / (evl + self.alpha))
    #import pdb; pdb.set_trace()
    self.alpha = 0.5 * gamma / E_W
    self.beta = 0.5 * (x.shape[0]*self.ny - gamma) / E_D
    
    print 'gamma = ' + str(gamma)
    print 'alpha = ' + str(self.alpha)
    print 'beta = ' + str(self.beta)
    
    #import pdb; pdb.set_trace()
            
  def train_bayes(self, x, t, alpha, beta, nouter = 3, ninner = 1, Nmax = 500, gtol = 1e-11):
    """"""
    # set initial values for alpha and beta
    self.alpha = alpha
    self.beta = beta
    for ko in range(nouter):
      self.train_BFGS(x, t, gtol = gtol, Nmax = Nmax, constrained = False)
      self.re_estimate(x, t)
      #import pdb; pdb.set_trace()
    
    