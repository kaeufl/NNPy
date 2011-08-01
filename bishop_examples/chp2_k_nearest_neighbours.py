import numpy as np
import matplotlib.pyplot as plt

N = 300 # number of samples
K = 70 # number of neighbours

weights = np.array([0.7, 0.3]) # weights of the gaussians
mu = np.array([0.3, 0.8]) # means of the gaussians
sigma = np.array([0.1, 0.1]) # std. deviations

# draw samples from a sum of two normal distirbutions
samples = np.append(np.random.normal(mu[0], sigma[0], N*weights[0]),
					np.random.normal(mu[1], sigma[1], N*weights[1]))
					
# pdf
x = np.arange(0,1,.01)
p = weights[0]*(1/np.sqrt(2*np.pi*sigma[0]**2))*np.exp(-(x-mu[0])**2 / (2*sigma[0]**2)) \
	+ weights[1]*(1/np.sqrt(2*np.pi*sigma[1]**2))*np.exp(-(x-mu[1])**2 / (2*sigma[1]**2))

########################################################################
# use a k nearest neighbours method to approximate the true pdf
########################################################################
# determine the minimum distance h to include K datapoints around point x

# calculate distances to x
dlist = np.sort(np.abs(
	np.tile(samples,(x.shape[0],1)) - np.tile(x,(N,1)).transpose()
	))
h = 2*dlist[:,K] # distance to Kth neighbour of x
# a density estimate is given by p(x)=K/NV
p_est = K / (N*h)

# plot a histogram of the samples
#plt.hist(samples,7)
#plt.hold(True)
#plt.plot(x,p)
#plt.show()

plt.figure()
plt.plot(x, p_est)
plt.plot(x, p)
plt.title(str(K)+'-nearest-neighbours estimation (N='+str(N)+')')
plt.show()




