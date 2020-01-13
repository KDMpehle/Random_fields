""" Test 2-D Karhunen- Loeve approximation: Exponential covariance"""
import numpy as np
from simulation import KL_2DNys
import matplotlib.pyplot as plt
# some tunable parameters
N = 100 # order of the KL expansion
n,m = (50,50) number of grid points
rho = 0.1 # scale length of the exponential
def Cov(x,y,rho = 0.1):
    r = np.sqrt(np.linalg.norm(x - y,2, axis =1)) # dist. between each point
    return np.exp(-r/rho)
lims = [0., 1., 0., 1.] # domain corners
X, phi, L = KL_2DNys(N,n,m,lims,Cov)
plt.loglog(range(N), L[:N]) # plot the eigenvalues descending.
plt.title("The exponential's first {} eigenvalues".format(N))
plt.show()
plt.close()
x,y = np.linspace(lims[0],lims[1],n), np.linspace(lims[2],lims[3],m)
xx,yy = np.meshgrid(x,y)# grid for plotting.
for i in range(6): # plot the first 6 eigenfunctions
    plt.subplot(2,3,i+1).set_title('i = {}'.format(i+1)) # add subplot
    e_func = np.array(phi[:,i]).reshape(n,m) # extract and reshape EFs
    plt.pcolor(xx,yy,e_func)
    plt.colorbar()
plt.show()
plt.close()
plt.pcolor(xx,yy,X) # plot the RF realisation
plt.title(r'2D exponential covariance with $\rho$ = {}'.format(rho)) 
plt.colorbar()
plt.show()
