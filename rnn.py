import numpy as np
from mlp import TLP

class RNN(TLP):
    def __init__(self, H = 3, d = 1, ny = 1, T = 100):
        """
        Create a fully-connected neural network with one hidden recurrent layer .
        @param H: number of hidden units
        @param ny: number of output units
        @param T: number of time-steps
        """
        TLP.__init__(self, H, d, ny)
        self.T = T
        z = np.zeros([T, H+1]) # hidden unit activations + bias
        self.z = z[None, :]
        self.z[:,:,0] = 1 # set extra bias unit to one
        
        # init recurrent weights
        self.wh = np.random.normal(loc=0.0, scale = 1,size=[H, H]) / np.sqrt(H) # TODO: check?
        
    def _check_inputs(self, x):
        if type(x) != np.array:
            x = np.array(x)
        if len(x.shape) != 3:
            x = x[None, :]
        if x.shape[2] != self.d:
            raise TypeError("Dimension of x should match number of input nodes.")
        return x
    
    def _forward(self, x, t, w1 = None, w2 = None, wh = None):
        if w1 == None:
            w1 = self.w1
        if w2 == None:
            w2 = self.w2
        if wh == None:
            wh = self.wh
        #t0=datetime.now()
        # calculate activation of hidden units
        # ordering hint: [training sample, time-step, hidden-unit, input]
        a = np.sum(w1[None, None,:, :] * x[:,t,None,:], axis = 3) + np.sum(wh[None, None, :, :] * self.z[:, t-1, :, None], axis = 2)
        self.z[:,t,1:] = self.g(a)
        # calculate output values
        y = np.dot(w2, self.z)
        #print 'eval of _tlp:' + str((datetime.now()-t0))
        return y
    
    # forward propagation
    def forward(self, x):
        x = self._check_inputs(x)
        # add an additional input for the bias
        x = np.append(np.ones([x.shape[0],1]),x,1)
        yn = self._forward(x)
        return yn.T
        
    def backward(self, x, dk, w2):
        """backward propagate the errors given by dk"""
        dj = (1 - self.z[1:]**2) * np.dot(w2.T[1:,:], dk)
        #t0=datetime.now()
        g1 = (dj[:,:,np.newaxis]*x[np.newaxis,:,:]).transpose(1,0,2)
        g2 = (dk[:,:,np.newaxis]*self.z.T[np.newaxis,:,:]).transpose(1,0,2)
        return g1, g2