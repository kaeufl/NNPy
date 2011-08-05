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
        if x.shape[1] != self.T:
            raise TypeError("Number of time-steps does not match.")
        return x
    
    def _forward(self, x, w1 = None, w2 = None, wh = None):
        if w1 == None:
            w1 = self.w1
        if w2 == None:
            w2 = self.w2
        if wh == None:
            wh = self.wh
        #t0=datetime.now()
        # calculate activation of hidden units
        # ordering hint: [training sample, time-step, hidden-unit, input]
        T = x.shape[1]
        for t in range(T):
            if t > 0:
                Z = self.z[:, t-1, 1:, None]
            else:
                Z = 0
            a = np.sum(w1[None, None,:, :] * x[:,t,None,:], axis = 3) + np.sum(wh[None, None, :, :] * Z, axis = 2)
            self.z[:,t,1:] = self.g(a)
            # calculate output values
        #y = np.dot(w2, self.z)
        y = np.sum(w2[None, None, :, :] * self.z[:,:,None, :], axis = 3)
        #print 'eval of _tlp:' + str((datetime.now()-t0))
        return y
    
    def forward(self, x):
        """forward propagate the given sequences through the network"""
        x = self._check_inputs(x)
        # add an additional input for the bias
        x = np.append(np.ones([x.shape[0], x.shape[1], 1]), x, 2)
        yn = self._forward(x)
        return yn.T
        
    def backward(self, x, dk, w2, wh):
        """backward propagate the errors given by dk"""
        # cycle backwards through time
        T = x.shape[1]
        # TODO: shift arrays instead of loop
        for t in range(T-1,-1,-1):
            if t < T-1:
                DJ = self.dj[:, t+1, :, None]
            else:
                DJ = 0
            self.dj[:, t, :] = self.dg(ga = self.z[:, t, :]) * (\
                    np.sum(w2[None, None, :, :] * dk[:,t,:, None], axis = 1) +\
                    np.sum(DJ * wh.T[None, None, :, :], axis = 2)
                )
        
        #t0=datetime.now()
        #TODO: shift arrays instead of loop
        for t in range(T):
            g1 = self.dj[:, t, :, None] * x[:, t, None, :]
            g2 = dk[:,t,:,None] * self.z[:, t, None, :]
            if t == 0:
                gh = 0
            else:
                gh = self.dj[:, t, :, None] * self.z[:, t-1, None, :]
        return g1, g2, gh