# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mlp import TLP

tlp = TLP(H = 8)

x = np.array([np.arange(-1,1,.01)])
sigma = 0.1
#x = np.random.normal(scale = sigma, size=(1,200)) # randomly sampled

#t = np.abs(x) # abs(x)
#t = np.sin(2*np.pi*x) # sin (x)
#t = np.sign(x) # step function
t = np.exp(-x**2/(2*sigma**2)) # gaussian

#plt.scatter(x.T, t.T, s = 5, c = 'k', edgecolors = 'k')
plt.plot(x.T, t.T)

#y = tlp.forward(x.T)
#plt.figure()
#plt.plot(x.T, y)
#plt.title('before training')

# on-line learning
#tlp.train(x.T,t.T,0.01, 1000, batch = False)
#tlp.train(x.T,t.T,0.1, 2000, batch = False)

# batch learning
tlp.train(x,t.T,0.001, 3000, batch = True)

#tlp.train_BFGS(x.T, t.T, 1e-1, 100)

y = tlp.forward(x.T)

plt.plot(x.T, y)
#plt.scatter(x.T, y, s = 5, c = 'g', edgecolors = 'g')

#plt.figure()
#plt.plot(tlp.Dw1)
#plt.plot(tlp.Dw2)
plt.show()
