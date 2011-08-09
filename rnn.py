import numpy as np
from mlp import TLP

class RNN(TLP):
    """
    TODO: * remove T, network should be able to handle input sequences of varying length
    """
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
        
        dj = np.zeros([T, H+1])
        self.dj = dj[None, :]
        
        # init recurrent weights
        self.wh = np.random.normal(loc=0.0, scale = 1,size=[H, H]) / np.sqrt(H) # TODO: check?
        self.Nwh = H**2
        
    def check_inputs(self, x):
        if type(x) != np.array:
            x = np.array(x)
        if len(x.shape) != 3:
            x = x[None, :]
        if x.shape[2] != self.d:
            raise TypeError("Dimension of x should match number of input nodes.")
        if x.shape[1] != self.T:
            raise TypeError("Number of time-steps does not match.")
        return x
    
    def unpack_weights(self, w):
        w1 = w[:self.Nw1].reshape([self.w1.shape[0],self.w1.shape[1]])
        w2 = w[self.Nw1:self.Nw1+self.Nw2].reshape([self.w2.shape[0],self.w2.shape[1]])
        wh = w[self.Nw1+self.Nw2:].reshape([self.H,self.H])
        return w1, w2, wh
    
    def pack_weights(self, w1, w2, wh):
        tmp = np.append(self.reshape_weights(w1), self.reshape_weights(w2), axis = 1)
        return np.append(tmp, self.reshape_weights(wh), axis = 1)
    
    def get_weights(self):
        return self.w1, self.w2, self.wh
    
    def set_weights(self, w1, w2, wh):
        self.w1 = w1
        self.w2 = w2
        self.wh = wh
    
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
        self.z = np.zeros([x.shape[0], T, self.H+1]) # hidden unit activations + bias
        self.z[:,:,0] = 1 # set extra bias unit to one
        for t in range(T):
            if t > 0:
                Z = self.z[:, t-1, 1:, None]
            else:
                Z = 0
            a = np.sum(w1[None, :, :] * x[:,t,None,:], axis = 2) + np.sum(wh[None, :, :] * Z, axis = 2)
            self.z[:,t,1:] = self.g(a)
            # calculate output values
        #y = np.dot(w2, self.z)
        y = np.sum(w2[None, None, :, :] * self.z[:,:,None, :], axis = 3)
        #print 'eval of _tlp:' + str((datetime.now()-t0))
        return y
    
    def prepare_inputs(self, x):
        """add an additional input for the biases"""
        return np.append(np.ones([x.shape[0], x.shape[1], 1]), x, 2)
        
    def backward(self, x, dk, w1, w2, wh):
        """backward propagate the errors given by dk"""
        # cycle backwards through time
        T = x.shape[1]
        # TODO: shift arrays instead of loop
        self.dj = np.zeros([x.shape[0], T, self.H+1])
        for t in range(T-1,-1,-1):
            if t < T-1:
                DJ = self.dj[:, t+1, 1:, None]
            else:
                DJ = np.zeros([x.shape[0], 1, self.H+1])
            self.dj[:, t, :] = self.dg(ga = self.z[:, t, :]) * (\
                    np.sum(w2[None, :, :] * dk[:,t,:, None], axis = 1) +\
                    np.append(np.zeros([x.shape[0], 1]), np.sum(DJ * wh.T[None, :, :], axis = 2), axis = 1)
                )
        
        #t0=datetime.now()
        #TODO: shift arrays instead of loop
        g1 = np.zeros([x.shape[0], self.H, self.d + 1])
        g2 = np.zeros([x.shape[0], self.ny, self.H + 1])
        gh = np.zeros([x.shape[0], self.H, self.H])
        for t in range(T):
            g1 = g1 + self.dj[:, t, 1:, None] * x[:, t, None, :]
            g2 = g2 + dk[:,t,:,None] * self.z[:, t, None, :]
            if t > 0:
                gh = gh + self.dj[:, t, 1:, None] * self.z[:, t-1, None, 1:]
        return g1, g2, gh