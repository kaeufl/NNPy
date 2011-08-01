# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

N = 1000 # no. of samples
it = 20 # no. of iterations

# initial configuration
M = 7 # no. of gaussian components
# create M random intitial means inside an annular region
mu = np.array([np.random.uniform(0.2, 0.6, M), np.random.uniform(0, 2*np.pi, M)])
sigma = np.tile(0.2, M) # std. deviations
P_j = np.tile(1.0/M, M) # prior component weights

# create random samples from a uniform annular distribution
samples = np.array([np.random.uniform(0.2,0.6, N), np.random.uniform(0, 2*np.pi, N)])

# convert everything to kartesian coordinates
mu = np.array([mu[0,:]*np.cos(mu[1,:]), mu[0,:]*np.sin(mu[1,:])])
samples = np.array([samples[0,:]*np.cos(samples[1,:]), samples[0,:]*np.sin(samples[1,:])])

# use the first M samples as starting values for mu
#mu = samples[:, 0:M]
#sigma = np.zeros(M)
#use the distance to the nearest neighbour component as starting value for sigma
#for m in range(M):
#  dmu = np.sort(np.sum((np.tile(mu[:,m], (M,1)) - mu.transpose())**2, 1)) #distance of mth mean to all other means
#  sigma[m] = dmu[1]
  
#print mu
#print sigma

#plot initial configuration
plt.figure()
plt.scatter(samples[0,:], samples[1,:])
plt.figure()
plt.scatter(mu[0,:], mu[1,:])
plt.xlim([-1,1])
plt.ylim([-1,1])
#plt.show()


for i in range(it):
	print 'Iteration: ' + str(i+1)
	# evaluate the posterior P(j|x^n)
	#p_x_j = np.zeros([N, M])
	L_j = np.zeros([N, M])
	P_x = np.zeros(N)
	P_post = np.zeros([N, M])
	
	for n in range(N):
		# calculate P(x^n|j) for each sample\
		d = np.tile(samples[:,n], (M,1))-mu.transpose()
		for j in range(M):
			L_j[n,j] = 1.0/(2*np.pi*sigma[j]**2) * np.exp(
				-(1.0/(2*sigma[j]**2)) * np.sum((samples[:,n] - mu[:,j])**2)
				)
		for j in range(M):
			P_x[n] = np.sum(L_j[n,:]*P_j[j])
			P_post[n,j] = L_j[n,j] * P_j[j] / P_x[n]

	# calculate error
	E = -1*np.sum(np.log(np.sum(L_j*P_j, 1)))
	print 'Error: ' + str(E)
	
	# update parameters
	mu = np.zeros([2, M])
	sigma = np.zeros(M)
	P_j = np.zeros(M)
	for j in range(M):
		mu[:,j] = np.sum(samples*P_post[:,j], 1) / np.sum(P_post[:,j])
		sigma[j] = np.sqrt(
			0.5 * np.sum(
				np.sum((samples.transpose() - np.tile(mu[:,j], (N, 1)))**2,1) \
				* P_post[:,j]
			) / np.sum(P_post[:,j])
		)
		P_j[j] = 1.0/N * np.sum(P_post[:,j])

fig = plt.figure()
# plot final distribution
x = np.arange(-1,1,.01)
y = np.arange(-1,1,.01)
X,Y = np.meshgrid(x,y)
Z = np.zeros(X.shape)
for j in range(M):
  Z = Z + P_j[j]*1.0/(2*np.pi*sigma[j]**2)*np.exp(
      -(1.0/(2*sigma[j]**2)) * ((X-mu[0,j])**2 + (Y-mu[1,j])**2)
    )
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z,rstride=4, cstride=4, cmap=cm.jet,
	        linewidth=0, antialiased=False)

#plt.scatter(mu[0,:],mu[1,:])
#plt.xlim([-1,1])
#plt.ylim([-1,1])
plt.show()

