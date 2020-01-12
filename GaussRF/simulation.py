import numpy as np
import scipy.sparse.linalg as spla
import numbers

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
    X: (M,) 1-D array of the random field
    phi: (M,N) 2-D arrray whose columns are the eigenfunctions
    L: (M,) 1-D array of the eigenvalues.
    """
    if N > M:
        raise ValueError(' Order of expansion N must be less than number of quadrature points')
    if not callable(Cov):
        raise TypeError('Cov must be a callable, bivariate function.')
    if quad == "EOLE":
        W = (1./M)*(b-a)*np.eye(M) # EOLE weight matrix.
        x1,x2 = np.meshgrid(x,x) # meshgrid to evaluate Covariance
        C = Cov(x1,x2) # Covariance matrix.
        B = np.dot(np.dot(np.sqrt(W),C),np.sqrt(W)) # symmetric B matrix.
        L,y = spla.eigsh(B, k=N) # first N eigens.
        arg_sort = np.argsort(-L) # indices for sorting
        L, y = L[arg_sort].real, y[:,arg_sort].real #eigens in descending order
        W_inv = np.sqrt((float(M)/(b-a)))*np.eye(M) #weights inverse sqrt matrix.
        phi = np.dot(W_inv,y) # eigenvecs of original problem.
        X = np.zeros(M) # preallocate solution array.
        Z = np.random.randn(M) # N(0,1) for KL exp.
        for i in range(N):
            X += Z[i]*np.sqrt(L[i])*phi[:,i]
        return X, phi, L
    else:
        raise ValueError("Only 'EOLE' quadrature is supported so far")

def KL_2DNys(N,n,m,lims,Cov,quad = "EOLE"):
    """
    Solver using the Nystrom method for finding the Karhunen-Loeve expansion
    -----
    input
    -----
    N: The order of the Karhunen-Loeve expansion.
    n,m: n and m are the number of gridpoints along x and y direction respectively. 
    lims: lims=[a,b,c,d] simulation domain is [a,b] x [c,d]
    Cov: the covariance function should. Should be given as c(x,y), x and y bivariate vectors.
    quad: The quadrature method used. EOLE will be the only implemented for now
    -----
    output
    -----
    X: (n,m) array of the RF simulation
    phi: (n*m, N) array holding eigenvectors discretised KL system
    L: (n*m,) array holding the eigenvalues
    """
    if N > n*m:
        raise ValueError("Order of expansion must be less than the number of quadrature points.")
    if not callable(Cov):
        raise TypeError("Cov must be a callable, bivariate function of 2-D vectors.")
    if quad == "EOLE":
        a,b,c,d = lims # extract domain limits
        A = (b-a)*(d-c) #Volume of domain.
        x,y = np.linspace(a,b,n) , np.linspace(c,d,m)
        W = (A/(n*m))*np.eye(n*m)# weights matrix.
        #create a list of coordinates
        x_pairs = np.hstack([np.repeat(x,m).reshape(n*m,1),np.tile(y,n).reshape(n*m,1)])#coordinate setup.
        x_mesh = np.hstack([np.repeat(xx,n*m,axis = 0), np.tile(xx,[n*m,1])]) # coordinate pairs grid for covariance.
        C = Cov(x_mesh[:,0:2],x_mesh[:,2:]).reshape(n*m,n*m) # Covariance matrix, looks to give correct eigens.
        B = np.dot(np.dot(np.sqrt(W),C),np.sqrt(W)) # symmetric pos. definite B
        L, y = spla.eigsh(B, k = N) # first N eigens of B
        arg_sort = np.argsort(-L)
        L,y = L[arg_sort].real, y[:,arg_sort].real #eigens in descending order
        W_inv = np.sqrt(float(n*m)/A)*np.eye(n*m) # inverse sqrt of weights
        phi = np.dot(W_inv,y) # eigenvecs of original problem.
        X = np.zeros((n,m)) # preallocate field.
        Z = np.random.randn(N) # N(0,1) for KL exp.
        for i in range(N): # The KL expansion
            X += np.sqrt(L[i])*Z[i]*phi[:,i].reshape(n,m)
        return X, phi, L
    else:
        raise ValueError("Only 'EOLE' quadrature is supported so far.")
def circ_embed1D(g,a,b,Cov):
    """
    The Circulant embedding method in 1-Dimension
    -----
    input
    -----
    g: exponent of the sample size N = 2^g
    a,b: terminals of domain.
    Cov: a stationary covariance function of one argument.
    -----
    output
    -----
    X: (N,) 1-D array of the random field
    """
    N = 2**g # sample size
    mesh = (b-a)/N # mesh size.
    x = np.arange(0,N)*mesh # domain grid
    r = np.zeros(2*N) # row defining the symmetric circulant matrix
    r[0:N] = Cov(x[0:N] - x[0])# first N entries of r row
    r[N+1:2*N] = Cov(x[N-1:0:-1] - x[0]) #last N-1 entries
    L  =np.fft.fft(r).real # eigenvalues of circulant matrix.
    neg_Evals = L[ L < 0] # produce a 'list' of negative eigenvalues
    if len(neg_vals) == 0:
        pass # if there are no negative eigenvalues, continue
    elif np.absolute(neg_Evals.min()) < 1e-16:
        L[L < 0] = 0 # eigenvalues are zero to machine tolerance.
    else:
        raise ValueError(" Could not find a positive definite embedding. Consider the KL method.")
    V1,V2 = np.random.randn(N), np.random.randn(N) # generate iid normals
    W = np.zeros(2*N, dtype = np.complex_)
    W[0] = np.sqrt(L[0]/(2*N))*V1[0]
    W[1:N] = np.sqrt(L[1:N]/(4*N))*(V1[1:N] +1j*V2[1:N])
    W[N] = np.sqrt(L[N]/(2*N))*V1[0]
    W[N+1:2*N] = np.sqrt(L[N+1:2*N]/(4*N))*(V1[N-1:0:-1] -1j*V2[N-1:0:-1])
    #Take fast Fourier transform of the special vector W
    w = np.fft.fft(W)
    return w[0:N].real # return first half of the vector.
def circ_embed2D(n,m,lims,Cov):
    """
    To simulate a 2-D stationary Gaussian field with the circulant embedding
    method in two dimensions.
    -----
    input
    -----
    n: number of grid points  in the x-direction.
    m: number of grid points in the y-direction.
    lims: a 4-d vector containing end points of rectangular domain.
    R: Covariance function of the Gaussian process, a bivariate function.
    -----
    output
    -----
    field1: The first field outputed, real part from the embedding method.
    field2: The second field outputed, imaginary part from the emedding method.
    """
    if not callable(Cov):
        raise TypeError("Cov must be a bivariate function")
    a,b,c,d = lims # extract interval terminals from lims variable.
    dx, dy = (b-a)/(m-1), (d-c)/(n-1) #increments in x- and y- directions.
    x,y = np.array(range(n),float)*dx, np.array(range(m),float)*dy # grid points.
    # Row is matrix of rows and Col matrix of columns defining block row.
    Row = Cov(x[None,:] - x[0], y[:,None] - y[0])
    Col = Cov(x[0] - x[None,:], y[:,None] - y[0])
    #Contruct Block row defining the circulant matrix.
    Block_R = np.vstack(
        [np.hstack([Row,Col[:,-1:0:-1]]),
         np.hstack([Col[-1:0:-1,:],Row[-1:0:-1,-1:0:-1]])])
    L = np.real(np.fft.fft2(Block_R))/((2*n-1)*(2*m-1)) # eigenvalues.
    neg_vals = L[L < 0] #produce a list of negative eigenvalues.
    if len(neg_vals) == 0:
        pass # if there are no negative Eigenvalues, continue
    elif np.absolute(np.min(neg_vals)) < 1e-16:
        L[L < 0] = 0 #EVs are zero to machine precision.
    else:
        raise ValueError(" Could not find a postive definite circulant embedding: Consider the KL method.")
    Z = (np.random.randn(2*m-1,2*n-1)
         + 1j*np.random.randn(2*m-1,2*n-1)) # N(0,1) grid
    W = np.fft.fft2(np.sqrt(L)*Z) # simulates N(0,C)
    return W[:m,:n].real, W[:m,:n].imag 
# Execution guard
#
if __name__ == "__main__":
    pass
