""" Test 1-D Karhunen-Loeve approximation: Brownian motion"""
import numpy as np
from simulation import KL_1DNys
import matplotlib.pyplot as plt
# some tunable parameters
N = 200 # order of the KL expansion
M = 200 # number of quadrature points
a, b = 0., 1. # interval endpoints
def Bm(s,t):# Brownian motion covariance
    return np.minimum(s,t)
X, phi, L = KL_1DNys(N,M,a,b,Bm)# simulation
#plot eigenvalues against exact: pi/L = (k-0.5)**2
L_exct = [(k + 0.5)**2 for k in range(N)] #+1 for python indexing
L_apprx = 1./(L[:N]*np.pi**2)
plt.plot(L_exct, label = "exact eigenvalues")
plt.plot(L_apprx,'x', label = "numerical eigenvalues")
plt.legend()
plt.ylabel(r' $\frac{1}{\lambda_k\pi^2}$')
plt.title(' Eigenvalues')
plt.show()
plt.close()
t= np.linspace(a,b,M)
#plot the first 6 eigenfunctions
for i in range(6):
    plt.subplot(2,3,i+1).set_title(r'$\phi_k$, k = {}'/format(i+1))
    exact = np.sqrt(2)*np.sin((k+0.5)*np.pi*t) # exact eigenfunc.
    apprx = np.abs(phi[:,k])*np.sign(exact) # approximate eigen, same sign as exact
    plt.plot(t, apprx,'x',label = "Numerical")
    plt.plot(t, exact, label = "exact")
    plt.legend()
plt.show()
