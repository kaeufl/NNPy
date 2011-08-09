# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mlp import TLP
from matplotlib import cm

N = 20**2

tlp = TLP(H = 5, d=2)

x = np.reshape(np.array([np.arange(-1,1,2/np.sqrt(N))]),(np.sqrt(N),1))
X,Y = np.meshgrid(x,x)

# create random samples from a 2d gaussian
#x2 = np.random.normal(scale = 0.05, size=(2,N))

T = 1/(2*np.pi) * np.exp(-1.0/(2*np.pi*0.05)**2 * (X**2 + Y**2))
#T2 = 1/(2*np.pi) * np.exp(-1.0/(2*np.pi*0.05)**2 * (x2[0]**2 + x2[1]**2))

x = np.reshape(np.array([X,Y]),[2, N]).T
t = np.reshape(T, [N, 1]).T

#tlp.train(x, t, 0.003, 1000, True)
#tlp.train(x, t, 0.00055, 1000, True)
#tlp.train(x2.T, T2, 0.1, 1000, False)
tlp.train_BFGS(x, t, 1e-5, 100)

z = tlp.forward(x)
#z2 = tlp.tlp(x2.T)
Z = np.reshape(z, [np.sqrt(N),np.sqrt(N)])

f = plt.figure()
ax = Axes3D(f)
#ax.contour(X,Y,Z)
ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
#ax.scatter(x2[0],x2[1],z2)

f = plt.figure()
ax = Axes3D(f)
ax.scatter(X,Y,t)
#ax.contour(X,Y,T)
#ax.plot_surface(X,Y,T, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
#ax.scatter(x2[0],x2[1],T2)

plt.figure()
#plt.plot(tlp.Dw1)
#plt.plot(tlp.Dw2)
plt.plot(tlp.E)
plt.show()


