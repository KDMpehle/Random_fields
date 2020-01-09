"""
This file contains prototype scripts for the approximate simulation
of 1-D and 2-D Gaussian random fields with a specified covariance function
C(x,y)
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
import time
from scipy import interpolate 
def KL_1DNys(N,M,a,b,Cov,quad = "EOLE"):
    """
    Karhunen-Loeve in 1-Dimension using Nystrom method.
    -----
    input
    -----
    N: Order of the Karhunen-Loeve expansion.
    M: number of quadrature intervals . N <=M
    a,b: domain of simulation, X_t for t in [a,b]
    Cov: The covariance function, a bivariate function
    quad: Quadrature used."EOLE" for the EOLE method. I tried Gauss-Legendre
    before and there was an issue with inaccurate simulation at the end
    points of the simulation domain
    -----
    output
    -----
    X: a 1-D array of the random field
    phi: a 2-D arrray whose columns are the eigenfunctions
    L: an 1-D array of the eigenvalues.
    """
    if N > M:
        raise ValueError('Order of expansion N should be less than quadrature\
points used')
    if quad == "EOLE": # EOLE method
        x = np.linspace(a,b,M+1) # EOLE uniform grid.
        W = (1./M)*(b-a)*np.eye(M+1) #EOLE weight matrix
        x1,x2 = np.meshgrid(x,x)
        C = Cov(x1,x2) # covariance matrix
        B = np.dot(np.dot(np.sqrt(W),C),np.sqrt(W)) #symmetric B matrix.
        L,y = spla.eigsh(B,k=N) #eigenvalues and vectors of B.
        arg_sort = np.argsort(-L) # indices for sorting.
        L,y =L[arg_sort].real, y[:,arg_sort].real #re-order the eigens.
        X = np.zeros(M+1)
        W_inv = np.sqrt((float(M)/(b-a)))*np.eye(M+1) # weights matrix.
        phi = np.dot(W_inv,y) # original eigenvector problem.
        Z = np.random.randn(M+1)
        for i in range(N):
            X += Z[i]*np.sqrt(L[i])*phi[:,i]
        return X, phi, L
    else:
        raise ValueError('We only have EOLE quadrature for now.')

def KL_2DNys(N,M,lims,Cov,quad = "EOLE"):
    """
    Simulate 2D Gaussian random fields with the Karhunen-Loeve approximation
    -----
    input
    -----
    N: The order of the Karhunen-Loeve expansion.
    M: M = [M1,M2] number of grid points along x and y direction.
    lims: lims = [a,b,c,d] simulation domain is [a,b] x [c,d]
    Cov: the covariance function. Should be given as c(x,y), x and y bivariate.
    quad: the quadrature method used. EOLE only implemented for now.
    """
    M1,M2 = M # extract M1 and M2
    n,m  = M1+1,M2+1 # save space. 
    a,b,c,d = lims # extract domain limits
    Om = (b-a)*(d-c) # Omega area of the rectangular domain.
    x, y = np.linspace(a,b,n), np.linspace(a,b,m) 
    W =(Om/(n*m))*np.eye(n*m)
    #create list of coordinates
    xx = np.hstack([np.repeat(x,m).reshape(n*m,1),np.tile(y,n).reshape(n*m,1)])
    xxx = np.hstack([np.repeat(xx,n*m,axis=0),np.tile(xx,[n*m,1])])
    C = Cov(xxx[:,0:2],xxx[:,2:]).reshape(n*m,n*m) #Covariance matrix, check this.
    B = np.dot(np.dot(np.sqrt(W),C),np.sqrt(W)) # symmetric pos def B
    # timing test
    t0 = time.clock()
    #L,y = np.linalg.eigh(B) # eigeevalues and vectors of B.
    L,y = spla.eigsh(B,k=N) #eigenvalues and vectors of B.
    arg_sort = np.argsort(-L)
    L,y =L[arg_sort].real, y[:,arg_sort].real #re-order the eigens.
    #Reverse order of EV and their vectors as eigh returns ascenidng order.
    #L,y = L[::-1], y[:,::-1]
    t1 = time.clock()
    print('Eigenvalue problem solved after: {} units'.format(t1-t0))
    W_inv = np.sqrt(float(n*m)/Om)*np.eye(n*m) # invert W matrix.
    phi = np.dot(W_inv,y)
    X = np.zeros((n,m)) # array to hold solution
    Z = np.random.randn(N) #iid standard normals
    for i in range(N):
        X+= np.sqrt(L[i])*Z[i]*phi[:,i].reshape(n,m)
    return X,phi,L # just return eigensuite for now
if __name__ == "__main__":
    test = "2D"
    if test == "1D":
        N = 200 # order of the KL expansion
        M = 200 # M+1 quadrature points
        def Bm(t,s):
            return np.minimum(t,s)
        a, b = 0., 1. # domain of simulation
        X,phi,L = KL_1DNys(N,M,a,b,Bm)
    # plot eigenvalues: pi/L = (k-0.5)**2 for BM
        L_ex = [(k+0.5)**2 for k in range(10)]
        L_app = 1./(L[:10]*np.pi**2)
        plt.plot(L_ex, label = "exact eigenvalues")
        plt.plot(L_app,'x', label = "numerical eigenvalues")
        plt.legend()
        plt.ylabel(r' $\frac{1}{\lambda_k\pi^2}$')
        plt.title(' Eigenvalues')
        plt.savefig("BM_EV_eg.pdf")
        plt.close()
        t= np.linspace(a,b,M+1) # t-grid
        exact = np.sqrt(2)*np.sin(4.5*np.pi*t) # exact fifth eigenfunction
        apprx= np.abs(phi[:,4])*np.sign(exact)# approximate 5th ef. Given same sign as exact.
        plt.plot(t, exact,'x',label= "Exact")
        plt.plot(t, apprx, label = "Numerical")
        plt.title("Eigenfunction, k = {}".format(5))
        plt.legend()
        plt.savefig("BM_EF_eg.pdf")
        plt.close()
        t = np.linspace(a,b,M+1) # time grid
        plt.plot(t,X)
        plt.title(" Brownian motion KL simulation")
        plt.savefig("BM_eg.pdf")
    elif test == "2D":
        N = 100 #KL expansion order
        M  =[50,50] # number of points in x- and y-directions.
        A = np.array([[1,0.8],[0.8,1]]) # anisotropic matrix
        #def Cov(x,y, A = A):
        #    s = x - y
        #    arg= A[0,0]*s[:,0]**2 +(A[1,0]+ A[0,1])*s[:,0]*s[:,1] + A[1,1]*s[:,1]**2
        #    return np.exp(-arg)
        def Cov(x,y,rho =0.1):
            r = np.linalg.norm(x - y,1,axis = 1)
            return np.exp(-r/rho)   
        lims = [0.,1.,0.,1.] # domain corners
        x,y = np.linspace(lims[0],lims[1],M[0]+1),np.linspace(lims[2],lims[3],M[1]+1)
        xx,yy = np.meshgrid(x,y, indexing ='ij')
        X,phi,L = KL_2DNys(N,M,lims,Cov)
        print(L[:3])
        plt.loglog(range(N),L[:N])
        plt.title("The exponential's first {} eigenvalues".format(N))
        plt.savefig("exponential_2D_eigenvalues.pdf")
        plt.close()
        for i in range(6):
            plt.subplot(2,3,i+1).set_title('k = {}'.format(i+1))
            e_func = np.array(phi[:,i]).reshape(M[0]+1,M[1]+1)
            plt.pcolor(xx,yy,e_func)
            plt.colorbar()
        plt.savefig("exponential_eigenfunctions.pdf")
        plt.close()
        #X = np.zeros((200,200)) # array to hold solution
        #Z = np.random.randn(N) #iid standard normals
        #s,t = np.linspace(0.,1.,200), np.linspace(0.,1.,200) # finer grid to evaluate on
        #ss,tt = np.meshgrid(s,t,indexing = 'ij')
        #for i in range(N):
        #    eig_array = np.array(phi[:,i]).reshape(M[0]+1,M[1]+1)
        #    e_func = interpolate.interp2d(x,y,eig_array)
        #    eig_field = e_func(s,t)
        #    X+= np.sqrt(L[i])*Z[i]*eig_field
        plt.pcolor(xx,yy,X)
        plt.colorbar()
        plt.savefig("exponential_RF_test.pdf")
        plt.show()
        
