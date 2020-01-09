import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

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
    X: a 1-D array of the random field
    """
    N = 2**g
    mesh = (b-a)/N
    #print(mesh)
    x = np.arange(0,N)*mesh # domain grid
    r = np.zeros(2*N) #row defining the symmetric circulant matrix
    r[0:N] = Cov(x[0:N]-x[0])
    r[N+1:2*N] = Cov(x[N-1:0:-1]-x[0])
    L =np.fft.fft(r).real # eigenvalues of circulant matrix.
    neg_Evals = L[L <0] # produce a 'list' of negative eigenvalues
    if len(neg_Evals) == 0:
        pass # if there are no negative eigenvalues, continues
    elif np.absolute(neg_Evals.min()) < 1e-16:
        L[ L <0] = 0 # eigenvalues are zero to machine precision.
    else:
        raise ValueError("Could not find a positive definite circulant matrix")
    V1,V2 = np.random.randn(N), np.random.randn(N) # generate iid normals
    W = np.zeros(2*N, dtype = np.complex_)
    W[0] = np.sqrt(L[0]/(2*N))*V1[0]
    W[1:N] = np.sqrt(L[1:N]/(4*N))*(V1[1:N] +1j*V2[1:N])
    W[N] = np.sqrt(L[N]/(2*N))*V1[0]
    W[N+1:2*N] = np.sqrt(L[N+1:2*N]/(4*N))*(V1[N-1:0:-1] -1j*V2[N-1:0:-1])
    #Take fast Fourier transform of the special vector W
    w = np.fft.fft(W)
    return w[0:N].real # return first half of the vector.
def circ_embed2D(n,m,lims,R):
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
    a,b,c,d = lims  #extract interval terminals from lims variable.
    dx,dy  = (b-a)/(n-1),(d-c)/(m-1)
    tx,ty = np.array(range(n),float)*dx, np.array(range(m),float)*dy
    Row, Col = np.zeros((n,m)), np.zeros((n,m))
    Row = R(tx[None,:] - tx[0],ty[:,None]-ty[0]) # Row definining block circulant
    Col = R(-tx[None,:] +tx[0], ty[:,None] - ty[0]) # columns defining block circulant
    #construct the block circulant matrix.
    Blk_R = np.vstack(
        [np.hstack([Row,Col[:,-1:0:-1]]),
         np.hstack([Col[-1:0:-1,:],Row[-1:0:-1,-1:0:-1]])])
    L = np.real(np.fft.fft2(Blk_R))/((2*m-1)*(2*n-1)) # eigenvalues
    neg_vals = L[L < 0]
    if len(neg_vals) == 0:
        pass # If there are no negative values, continue
    elif np.absolute(np.min(neg_vals)) < 10e-15:
        L[ L < 0] = 0 # EV negative due to numerical precision, set to zero.
    else:
        raise ValueError(" Could not find a positive definite embedding")
    L = np.sqrt(L) # component wise square root
    Z = (np.random.randn(2*m -1,2*n-1)
         + 1j*np.random.randn(2*m-1,2*n-1)) # add a standard normal complex
    F = np.fft.fft2(L*Z)
    F = F[:m,:n]
    field1, field2 = np.real(F), np.imag(F)
    return field1, field2
        
    raise ValueError("Could not find a positive definite embedding")
    lam = np.sqrt(lam)
test = "2D" # set variable for testing the circulant embedding algorithm.                        
if __name__ == "__main__":
    if test == "1D":
        h=0.01 # Hurst parameter for fractional Brownian motion. 
        l = 1.0 # parameter for the exponential covariance
        g= 10
        N = 2**g 
        #t_vals = np.linspace(0,1.,N+1)
        t_vals = np.linspace(-5.,5.,N)
        def Cov(k,H=h):
            return (np.abs(k-1)**(2*H) -2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))/2
        def exp_cov(r, l =l):
            return np.exp(-r/l)
        X = circ_embed1D(g,-5.,5.,exp_cov)
        #X = circ_embed1D(g,0.,float(N),Cov) # simulate a fractional Gaussian noise.
        #X2 = np.insert((1./N)**h*np.cumsum(X),0,0) # fractional Brownian motion + starting point
        plt.plot(t_vals,X)
        plt.xlim([-5,5])
        #plt.title(" H = {:.2f}".format(h))
        #plt.savefig("circ_embed_fBm.pdf")
        plt.title(" Exponential covariance, scale length l = {:.1f}".format(l))
        plt.savefig("circ_embed1D_exp.pdf")
        plt.show()
    elif test == "2D":
        l1 = 15 # scale length in x/y direction
        l2 = 50 # scale length in x/y direction
        def R(x,y, l1 = l1, l2 = l2):
            A  = np.array([[3,1],[1,2]])
            arg = ( (x/l1)**2*A[1,1] + (A[0,1] + A[1,0])*(x/l1)*(y/l2)
                    + (y/l2)**2*A[0,0] )
            return np.exp(-np.sqrt(arg))
        lims = [1.,383.,1.,511.] # limits: dx = 1, dy =1
        field1, field2 = circ_embed2D(383,511,lims,R)
        plt.title(r' $l_1$ = {}, $l_2$ = {}'.format(l1,l2))
        plt.imshow(field1)
        plt.savefig('circ_embed2D_aniso.pdf')    
        plt.show()
