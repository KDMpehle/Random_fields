""" Test 1-D circulant embedding: Simulate an exponential covariance."""
import numpy as np
from simulation import circ_embed1D
import matplotlib.pyplot as plt
#Some tunable parameters
l = 1. # scale length of the exponential covariance
g = 10 # exponent of N = 2^g for the sample size.
a, b = -5. , 5. # interval endpoints.
#The Exponential covariance.
def exp_cov(r, l = l):
    return np.exp(-r/l)
X = circ_embed1D(g,a,b,exp_cov)
t_vals = np.linspace(a,b,N) # The mesh points.
plt.plot(t_vals,X) # plot the solution
plt.title("1D exp covariance with scale l = {:.1f}".format(l))
plt.show()
