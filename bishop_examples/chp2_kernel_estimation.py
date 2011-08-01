import numpy as np
import matplotlib.pyplot as plt

N = 30 # number of samples
h = 0.01 # width parameter of the kernel function

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
# use a normal kernel function to estimate the density from the N samples
########################################################################
# calculate values of the kernel function H(x) and sum over the data set
H = np.zeros([N, x.size])
for k in range(samples.size):
	xn = samples[k]
	H[k, :] = (1/np.sqrt(2*np.pi*h**2)) * np.exp( -(1/(2*h**2)) * (np.abs(x-xn)**2) )
p_est = np.sum(H,0)/N

# Kullback-Leibler distance
L = -np.sum(p*np.log(p_est/p))
print 'Kullback-Leibler distance: L='+str(L)

# plot a histogram of the samples
#plt.hist(samples,7)
#plt.hold(True)
#plt.plot(x,p)
#plt.show()

plt.figure()
plt.plot(x, p_est)
plt.plot(x, p)
plt.title('Gaussian kernel estimation (h='+str(h)+', N='+str(N)+')')
plt.show()




