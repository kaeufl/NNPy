# -*- coding: utf-8 -*-
import numpy as np
#from datetime import datetime

class TLP:
    """Two layer perceptron"""
    def __init__(self, H = 3, d = 1, ny = 1, linear_output = True, 
                             error_function = 'sum_of_squares', debug_output = False):
        self.H = H
        self.d = d
        self.ny = ny
        self.linear_output = linear_output
        self.En = getattr(self, 'E_' + error_function)
        self.dEn = getattr(self, 'dE_'+error_function)
        self.HEn = getattr(self, 'HE_'+error_function)
        self.debug_output = debug_output
        self.error_function = error_function
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
        self.E = []
    
    def init_weights(self, alpha1, alpha2):
        """Init weights according to Gaussian prior"""
        self.w1 = np.random.normal(loc=0.0, scale = 1.0/np.sqrt(alpha1),size=[self.H, self.d+1])
        self.w2 = np.random.normal(loc=0.0, scale = 1.0/np.sqrt(alpha2),size=[self.ny, self.H+1])
        
    def g(self, a):
        """activation function"""
        return np.tanh(a)
    
    def dg(self, a = None, ga = None):
        """derivative of activation function"""
        if a:
            return 1 - self.g(a)**2
        else:
            return 1 - ga**2
    
    def En(self, y, t):
        """error function"""
        pass
    
    def dEn(self, x, y, t):
        """derivatives of error function"""
        pass
    
    def HEn(self, x, y, t):
        """Hessian of error function"""
        pass
    
    def E_sum_of_squares(self, y, t, *w):
        """Sum-of-squares error function"""
        #return 0.5 * np.sum((y - t.T)**2, 0)
        return 0.5 * np.sum((y - t)**2, 0)
        
    def dE_sum_of_squares(self, x, y, t, *w):
        """Derivatives of sum-of-squares error function"""
        #if w2 == None:
        #    w2 = self.w2
        # dk = y - t.T
        dk = y - t
        return self.backward(x, dk, *w)
        
    def HE_sum_of_squares(self, x, y, t, w1 = None, w2 = None):
        """Returns the Hessian of the error function with respect to the weights"""
        if w1 == None:
            w1 = self.w1
        if w2 == None:
            w2 = self.w2
        
        # consistency check
        if x.shape[1] != self.d + 1:
            raise TypeError("x doesn't have the right dimension, add an additional column for the first layer biases first.")
        
        # consequently evaluate v^T H for a complete set of unit vectors
        V = np.eye(w1.size + w2.size)
        H = np.zeros([w1.size + w2.size, w1.size + w2.size])
        for v in range(V.shape[0]):
            RdEnw1, RdEnw2 = self._Hdotv(x, y, t, V[v], w1, w2)
            H[:, v] = np.append(RdEnw1, RdEnw2, axis = 1)
        return H

    def _Hdotv(self, x, y, t, v, w1, w2):
        """Calculate the Hessian times a vector v as descirbed in Bishop 1995, 4.10.7
        v is a vector of length NW (number of weights and biases)
        """
        # derivatives of activation function with respect to the weights (without biases)
        dg = 1 - self.z[1:]**2
        ddg = -2.0*self.z[1:]*dg
        
        v_w1 = np.reshape(v[0:w1.size], w1.shape)
        v_w2 = np.reshape(v[w1.size:], w2.shape)
        
        # the operator R = v^T\nabla is introduced and v^T\nabla w = v is used
        # forward propagate R(.)
        Raj = np.sum(v_w1[None,:,:] * x[:,None,:], axis = 2)
        Rzj = dg.T * Raj
        Ryk = np.sum(w2[None, :,1:] * Rzj[:, None, :], axis = 2) \
                    + np.sum(v_w2[None,:,:] * self.z.T[:,None,:], axis = 2)
        
        # back-propagate R(.)
        dk = y - t.T
        dj = dg.T * np.sum(w2[None, :, 1:] * dk[:, :, None], axis = 1)
        Rdk = Ryk
        Rdj = ddg.T * Raj * np.sum(w2[None, :, 1:] * dk[:, :, None], axis = 1) \
                    + dg.T * np.sum(v_w2[None, :, 1:] * dk[:, :, None], axis = 1) \
                    + dg.T * np.sum(w2[None, :, 1:] * Rdk[:, :, None], axis = 1)
        
        # get second derivatives by evaluation of R(dEnw1) and R(dEnw2)
        RdEnw1 = x[:, None, :] * Rdj[:, :, None]
        # we have to treat the bias terms different from the weight terms
        RdEnw2 = Rdk[:, :, None] * self.z.T[:, None, :] \
                         + np.append(np.zeros([x.shape[0], self.ny, 1]), dk[:, :, None] * Rzj[:,None,:], axis = 2)
        # sum over patterns
        RdEnw1 = np.sum(RdEnw1, axis=0)
        RdEnw2 = np.sum(RdEnw2, axis=0)
        return (np.reshape(RdEnw1, [RdEnw1.shape[0]*RdEnw1.shape[1]]), \
                        np.reshape(RdEnw2, [RdEnw2.shape[0]*RdEnw2.shape[1]]))
    
    def deriv(self, x):
        """Return the derivative of the network outputs with respect to the weights evaluated at w"""
        # do a forward propagation to update the z
        y = self._forward(x)
        dydw1 = np.zeros([x.shape[0], self.ny, self.w1.shape[0], self.w1.shape[1]])
        dydw2 = np.zeros([x.shape[0], self.ny, self.w2.shape[0], self.w2.shape[1]])
        
        for k in range(self.ny):
            dk = np.zeros([self.ny, x.shape[0]])
            dk[k, :] = 1
            dydw1[:, k], dydw2[:, k] = self.backward(x, dk, self.w2)
        return dydw1, dydw2
        
    def softmax(self, x):
        # prevent overflow
        maxval = np.log(np.finfo(float).max) - np.log(x.shape[0])
        x = np.minimum(maxval, x)
        # prevent underflow
        minval = np.finfo(float).eps
        x = np.maximum(minval, x)
        return np.exp(x) / np.sum(np.exp(x), axis = 0)
    
    def _forward(self, x, w1 = None, w2 = None):
        if w1 == None:
            w1 = self.w1
        if w2 == None:
            w2 = self.w2
        #t0=datetime.now()
        # calculate activation of hidden units and add an additonal element as input for the bias
        self.z = np.append(np.ones([1, x.shape[0]]), self.g(np.dot(w1, x.T)), axis = 0)
        # calculate output values
        y = np.dot(w2, self.z)
        #print 'eval of _tlp:' + str((datetime.now()-t0))
        return y
        
    def check_inputs(self, x):
        if type(x) != np.array:
            x = np.array(x)
        if len(x.shape) != 2:
            x = np.array([x])
        if x.shape[1] != self.d:
            raise TypeError("Dimension of x should match number of input nodes.")
        return x
    
    # forward propagation
    def forward(self, x):
        x = self.check_inputs(x)
        x = self.prepare_inputs(x)
        yn = self._forward(x)
        return yn.T
        
    def backward(self, x, dk, w1 = None, w2 = None):
        """backward propagate the errors given by dk"""
        if w2 == None:
            w2 = self.w2
        dj = (1 - self.z[1:]**2) * np.dot(w2.T[1:,:], dk)
        #t0=datetime.now()
        g1 = (dj[:,:,np.newaxis]*x[np.newaxis,:,:]).transpose(1,0,2)
        g2 = (dk[:,:,np.newaxis]*self.z.T[np.newaxis,:,:]).transpose(1,0,2)
        return g1, g2

#    # train the network via fixed step gradient descent back-propagation
#    def train_fsgd(self, x, t, eta, nt, batch = False):
#        x = self.check_inputs(x)
#        if type(t) != np.array:
#            t = np.array(t)
#        # add an additional input for the biases
#        x = self.prepare_inputs(x)
#        
#        for it in range(nt):
#            print "epoch: " + str(it)
#            dEw1 = 0
#            dEw2 = 0
#            E = 0
#            y = self._tlp(x)
#            for xi in range(x.shape[0]):
#                # calculate activations for current sample
#                #y = self._tlp(x[xi])
#                # calculate derivatives
#                dEnw1, dEnw2 = self.dEn(x[xi], y[:,xi], t[xi])
#                dEw1 = dEw1 + dEnw1
#                dEw2 = dEw2 + dEnw2
#                E = E + self.En(y[:,xi], t[xi])
#                # perform weight updates
#                if not batch: # on-line learning
#                    Dw1 = - eta * dEnw1
#                    Dw2 = - eta * dEnw2
#                    self.w1 = self.w1 + Dw1
#                    self.w2 = self.w2 + Dw2
#            if batch:
#                    Dw1 = - eta * dEw1
#                    Dw2 = - eta * dEw2                        
#                    self.w1 = self.w1 + Dw1
#                    self.w2 = self.w2 + Dw2
#            # keep \Delta ws and E to monitor convergence
#            self.Dw1.append(np.abs(np.sum(dEw1)));
#            self.Dw2.append(np.abs(np.sum(dEw2)));
#            print 'E = ' + str(E)
#            if self.debug_output:
#                print '1st layer weights:'
#                print self.w1
#                print '2nd layer weights:'
#                print self.w2
#            self.E.append(E)
#        print 'residual error: ' + str(E)
        
    def reshape_weights(self, w):
        if len(w.shape) == 3:
            return w.reshape([w.shape[0], w.shape[1]*w.shape[2]])
        else:
            return w.flatten()
        
    def unpack_weights(self, w):
        w1 = w[:self.Nw1].reshape([self.w1.shape[0],self.w1.shape[1]])
        w2 = w[self.Nw1:].reshape([self.w2.shape[0],self.w2.shape[1]])
        return w1, w2
    
    def pack_weights(self, w1, w2):
        return np.append(self.reshape_weights(w1), self.reshape_weights(w2), axis = 1)
    
    def get_weights(self):
        return self.w1, self.w2
    
    def set_weights(self, w1, w2):
        self.w1 = w1
        self.w2 = w2
        
    def prepare_inputs(self, x):
        """add an additional input for the biases"""
        return np.append(np.ones([x.shape[0],1]),x,1)
    
    def train_BFGS(self, x, t, gtol = 1e-2, Nmax = 1000, constrained = False,
                                 callback = None):
        """train network using the Broyden-Fletcher-Goldfarb-Shanno quasi-Newton method"""
        from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b
        from scg import scg
        from datetime import datetime
        
        # objective function to be minimized, takes a weight vector and returns an error measure
        def f(w, x, t):
            #t0=datetime.now()
            weights = self.unpack_weights(w)
            y = self._forward(x, *weights)
            E = np.sum(self.En(y, t, *weights))
            self.E.append(E)
            # store current network output for internal use
            self._y = y
            #print 'eval of f:' + str((datetime.now()-t0))
            return E
        
        # gradient of f
        def df(w, x, t):
            #t0=datetime.now()
            weights = self.unpack_weights(w)
            y = self._forward(x, *weights)
            dEnw = self.pack_weights(*self.dEn(x, y, t, *weights))
            g = np.sum(dEnw, 0)
            #print 'eval of df:' + str((datetime.now()-t0))
            return g
            
        def iter_status(xk):
            self._t1 = datetime.now()-self._t0
            self._iteration_no = self._iteration_no + 1
            print 'Iteration: ' + str(self._iteration_no)
            print 'E = ' + str(self.E[-1])
            print 'execution time: ' + str(self._t1)
            self._t0 = datetime.now()
            
        if callback == None: callback = iter_status
        
        t0=datetime.now()
        x = self.check_inputs(x)
        if type(t) != np.array:
            t = np.array(t)
        
        x = self.prepare_inputs(x)
        
        w = self.pack_weights(*self.get_weights())
        if not constrained:
            self._iteration_no = 0
            self._t0 = datetime.now()
            w_new = fmin_bfgs(f, w, df, (x, t), gtol = gtol, maxiter = Nmax, callback = callback)
            #w_new = fmin_cg(f, w, df, (x, t), gtol = gtol, maxiter = Nmax)
        else:
            #[w_new, E_min, d] = fmin_l_bfgs_b(f, w, df, (x, t), bounds=((-100, 100),)*w.shape[0],
                                                                                #approx_grad=False, factr = 1e7, pgtol = gtol,
                                                                                #maxfun = Nmax)
            #print d['task']
            tmp = scg(w, f, df, x,t, 
                                xPrecision=np.finfo(float).eps, 
                                nIterations=Nmax, 
                                fPrecision=np.finfo(float).eps
                                )
            w_new = tmp['x']
            print tmp['reason']
        #w_new = leastsq(f, w, (x, t), df)
        self.set_weights(*self.unpack_weights(w_new))
        
        print 'Training complete, took ' + str(datetime.now()-t0) + 's'