""" Test 2-D circulant embedding: (1-x)*exp(-x) type covariance"""
import numpy as np
from simulation import circ_embed2D
import matplotlib.pyplot as plt
#Some tunable parameters
l1 = 50 # scale length in the x-direction
l2 = 15 # scale length in the y-direction
n = 511 # number of points in x
m = 383 # number of points in y
def Cov(x,y,l1=l1,l2 = l2):
    A = np.array([[3,1],[1,2]]) # Anisotropic matrix
    arg = ( (x/l1)**2*A[0,0] + (A[0,1] + A[1,0])*(x/l1)*(y/12)
            + (y/l2)**2*A[1,1] )
    return np.exp(-np.sqrt(arg))
lims = [1., float(n), 1., float(m)] #unit increments dx= 1, dy = 1
field1, field2 = circ_embed2D(n,m,lims, Cov)
plt.title(r' Anistropic exampl, $l_1$ = {}, $l_2$ = {}'.format(l1,l2))
plt.imshow(field, origin = 'lower')
plt.show()
