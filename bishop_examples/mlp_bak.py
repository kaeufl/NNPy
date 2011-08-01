# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime
from matplotlib import cm, mlab
import matplotlib.pyplot as plt

class TLP:
  def __init__(self, H = 3, d = 1, ny = 1, linear_output = True, 
               error_function = 'sum_of_squares', debug_output = False):
    self.H = H
    self.d = d
    self.ny = ny
    self.linear_output = linear_output
    self.En = getattr(self, 'E_' + error_function)
    self.dEn = getattr(self, 'dE_'+error_function)
    self.debug_output = debug_output
    if (error_function == 'mdn' and linear_output == False):
      raise 'Linear output units are required for the use of the mixture density network error function "mdn".'
    
    # init weights randomly
    #self.w1 = np.random.uniform(-1.0, 1.0, size=[H, d+1]) # 1st layer weights + bias
    #self.w2 = np.random.uniform(-1.0, 1.0, size=[ny, H+1]) # 2nd layer weights + bias
    #self.w1 = np.random.normal(loc=0.0, scale = (d+1)**(-0.5),size=[H, d+1]) # 1st layer weights + bias
    #self.w1 = np.random.normal(loc=0.0, scale = (d)**(-0.5),size=[H, d+1]) # 1st layer weights + bias
    #self.w2 = 1.0/(ny*(H+1))*np.random.uniform(-1.0, 1.0, size=[ny, H+1]) # 2nd layer weights + bias
    # taken from netlab mlp.m:
    self.w1 = np.random.normal(loc=0.0, scale = 1,size=[H, d+1])/np.sqrt(d+1) # 1st layer weights + bias
    self.w2 = np.random.normal(loc=0.0, scale = 1,size=[ny, H+1])/np.sqrt(H+1) # 2nd layer weights + bias
    
    self.Nw1 = H * (d+1)
    self.Nw2 = ny * (H+1)
    self.z = np.zeros(self.H+1) # hidden unit activations + bias
    
    # performance information
    self.Dw1 = []
    self.Dw2 = []
    self.E = []
    
  def g(self, a):
    """activation function"""
    return np.tanh(a)
  
  def En(self, y, t):
    """error function"""
    pass
  
  def dEn(self, x, y, t):
    """derivatives of error function"""
    pass
  
  def E_sum_of_squares(self, y, t):
    """Sum-of-squares error function"""
    return 0.5 * np.sum((y - t)**2)
    
  def dE_sum_of_squares(self, x, y, t, w2 = None):
    """Derivatives of sum-of-squares error function"""
    if w2 == None:
      w2 = self.w2
    if self.linear_output:
      dk = y - t
    #else:
      #dk = (y-t) * y * (1-y)
    #dj = self.z[1:] * (1 - self.z[1:]) * np.dot(w2.T[1:,:], dk).T
    dj = (1 - self.z[1:]**2) * np.dot(w2.T[1:,:], dk).T
    dEnw1 = np.outer(dj, x)
    dEnw2 = np.outer(dk, self.z)
    return dEnw1, dEnw2
  
  def softmax(self, x):
    # prevent overflow
    maxval = np.log(np.finfo(float).max) - np.log(x.shape[0])
    x = np.minimum(maxval, x)
    # prevent underflow
    minval = np.finfo(float).eps
    x = np.maximum(minval, x)
    return np.exp(x) / np.sum(np.exp(x))
  
  def _tlp(self, x, w1 = None, w2 = None):
    if w1 == None:
      w1 = self.w1
    if w2 == None:
      w2 = self.w2
    # calculate activation of hidden units and add an additonal element as input for the bias
    self.z = np.append(np.array([1]), self.g(np.dot(w1, x)))
    # calculate output values
    if self.linear_output:
        y = np.dot(w2, self.z)
    else:
        y = self.g(np.dot(w2, self.z))
    return y
    
  def _check_inputs(self, x):
    if type(x) != np.array:
      x = np.array(x)
    if len(x.shape) != 2:
      x = np.array([x])
    if x.shape[1] != self.d:
      raise TypeError("Dimension of x should match number of input nodes.")
    return x
  
  # forward propagation
  def forward(self, x):
    x = self._check_inputs(x)
    # add an additional input for the bias
    #x = np.append(np.tile([1],(x.shape[0],1)),x,1)
    x = np.append(np.ones([x.shape[0],1]),x,1)
    yn = np.zeros([x.shape[0], self.ny])
    for xi in range(x.shape[0]):
        yn[xi,:] = self._tlp(x[xi])
    return yn

  # train the network via fixed step gradient descent back-propagation
  def train(self, x, t, eta, nt, batch = False):
    x = self._check_inputs(x)
    if type(t) != np.array:
      t = np.array(t)
    # add an additional input for the biases
    x = np.append(np.tile([1],(x.shape[0],1)),x,1)
    
    for it in range(nt):
        print "epoch: " + str(it)
        dEw1 = 0
        dEw2 = 0
        E = 0
        for xi in range(x.shape[0]):
          # calculate activations for current sample
          y = self._tlp(x[xi])
          # calculate derivatives
          dEnw1, dEnw2 = self.dEn(x[xi], y, t[xi])
          dEw1 = dEw1 + dEnw1
          dEw2 = dEw2 + dEnw2
          E = E + self.En(y, t[xi])
          # perform weight updates
          if not batch: # on-line learning
            Dw1 = - eta * dEnw1
            Dw2 = - eta * dEnw2
            self.w1 = self.w1 + Dw1
            self.w2 = self.w2 + Dw2
        if batch:
            Dw1 = - eta * dEw1
            Dw2 = - eta * dEw2            
            self.w1 = self.w1 + Dw1
            self.w2 = self.w2 + Dw2
        # keep \Delta ws and E to monitor convergence
        self.Dw1.append(np.abs(np.sum(dEw1)));
        self.Dw2.append(np.abs(np.sum(dEw2)));
        print 'E = ' + str(E)
        if self.debug_output:
          print '1st layer weights:'
          print self.w1
          print '2nd layer weights:'
          print self.w2
        self.E.append(E)
    print 'residual error: ' + str(E)
  
  def train_BFGS(self, x, t, gtol = 1e-2, Nmax = 1000, constrained = False):
    """train network using the Broyden-Fletcher-Goldfarb-Shanno quasi-Newton method"""
    from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b
    from scg import scg
    from datetime import datetime
    
    # objective function to be minimized, takes a weight vector and returns an error measure
    def f(w, x, t):
      #t0=datetime.now()
      w1 = w[:self.Nw1].reshape([self.w1.shape[0],self.w1.shape[1]])
      w2 = w[self.Nw1:].reshape([self.w2.shape[0],self.w2.shape[1]])
      E = 0
      for xi in range(x.shape[0]):
        y = self._tlp(x[xi], w1, w2)
        E = E + self.En(y, t[xi])
      self.E.append(E)
      #print 'eval of f:' + str((datetime.now()-t0).microseconds)
      #print 'E=' + str(E)
      return E
    
    # gradient of f
    def df(w, x, t):
      #t0=datetime.now()
      w1 = w[:self.Nw1].reshape([self.w1.shape[0],self.w1.shape[1]])
      w2 = w[self.Nw1:].reshape([self.w2.shape[0],self.w2.shape[1]])
      g = 0
      for xi in range(x.shape[0]):
        y = self._tlp(x[xi], w1, w2)
        dEnw1, dEnw2 = self.dEn(x[xi], y, t[xi], w2)
        g = g + np.append(_reshape_w(dEnw1), _reshape_w(dEnw2))
      
      #print 'eval of df:' + str((datetime.now()-t0).microseconds)
      return g
      
    def iter_status(xk):
      self._iteration_no = self._iteration_no + 1
      print 'Iteration: ' + str(self._iteration_no)
      print 'E = ' + str(self.E[-1])
    
    def _reshape_w(w):
      return w.reshape([w.shape[0]*w.shape[1]])
    
    t0=datetime.now()
    x = self._check_inputs(x)
    if type(t) != np.array:
      t = np.array(t)
    # add an additional input for the biases
    x = np.append(np.tile([1],(x.shape[0],1)),x,1)
    
    w = np.append(_reshape_w(self.w1), _reshape_w(self.w2))
    if not constrained:
      self._iteration_no = 0
      w_new = fmin_bfgs(f, w, df, (x, t), gtol = gtol, maxiter = Nmax, callback = iter_status)
      #w_new = fmin_cg(f, w, df, (x, t), gtol = gtol, maxiter = Nmax)
    else:
      #[w_new, E_min, d] = fmin_l_bfgs_b(f, w, df, (x, t), bounds=((-100, 100),)*w.shape[0],
                                        #approx_grad=False, factr = 1e7, pgtol = gtol,
                                        #maxfun = Nmax)
      #print d['task']
      tmp = scg(w, f, df, x,t, xPrecision=np.finfo(float).eps, nIterations=Nmax)
      w_new = tmp['x']
      print tmp['reason']
    #w_new = leastsq(f, w, (x, t), df)
    self.w1 = w_new[:self.Nw1].reshape([self.w1.shape[0],self.w1.shape[1]])
    self.w2 = w_new[self.Nw1:].reshape([self.w2.shape[0],self.w2.shape[1]])
    
    print 'Training complete, took ' + str(datetime.now()-t0) + 's'


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
    
  def init_weights(self, t, prior):
    """
    initialze weights and biases so that the network models the unconditional density 
    of the target data p(t)
    t: target data
    prior: 1/sigma^2
    """
    from scipy.cluster.vq import kmeans2
    from scipy.spatial.distance import cdist
    sigma = np.sqrt(1.0/prior)
    # init weights from gaussian with width given by prior
    self.w1 = np.random.normal(loc=0.0, scale = 1,size=[self.H, self.d+1])*sigma # 1st layer weights + bias
    self.w2 = np.random.normal(loc=0.0, scale = 1,size=[self.ny, self.H+1])*sigma # 2nd layer weights + bias
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
    """Returns the parameters of the gaussian mixture."""
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
    M = int(self.M)
    # distance between target data and gaussian kernels    
    dist = np.sum((T-mu)**2, 1)
    phi = (1.0 / (2*np.pi*sigma)**(0.5*self.c)) * np.exp(- 1.0 * dist / (2 * sigma))
    # prevent underflow
    return np.maximum(phi, np.finfo(float).eps)
    
  def E_mdn(self, y, t):
    """mdn error function"""
    M = int(self.M)
    alpha, sigma, mu = self.getMixtureParams(y)
    
    #T = np.tile(t, [M,1])
    phi = self._phi(t, mu, sigma)
    probs = np.maximum(np.sum(alpha * phi), np.finfo(float).eps)
    return - np.log(probs)
  
  def dE_mdn(self, x, y, t, w2 = None):
    """derivative of mdn error function"""
    if w2 == None:
      w2 = self.w2
    
    M = int(self.M)
    # avoid underrun
    alpha, sigma, mu = self.getMixtureParams(y)
    phi = self._phi(t, mu, sigma)
    
    pi = alpha * phi / np.sum(alpha * phi)
    
    # derivatives of E with respect to the output variables (s. Bishop 1995, chp. 6.4)
    dE_dy_alpha = alpha - pi
    dE_dy_sigma = - 0.5 * pi * ((np.sum((t-mu)**2 , 1) / sigma) - self.c)
    dE_dy_mu = pi * (mu - t).T / sigma

    dk = np.zeros(self.ny)
    dk[0:M] = dE_dy_alpha
    dk[M:2*M] = dE_dy_sigma
    dk[2*M:] = np.reshape(dE_dy_mu, [M*self.c])
    
    # back-propagate the dks
    dj = (1 - self.z[1:]**2) * np.dot(w2.T[1:,:], dk).T
    # evaluate derivatives with respect to the weights
    dEnw1 = np.outer(dj, x)
    dEnw2 = np.outer(dk, self.z)
    return dEnw1, dEnw2
    
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
  