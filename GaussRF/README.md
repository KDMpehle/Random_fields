GaussRF
--------
 A python package for numerical simulation of Gaussian random fields.

Overview
------
GaussRF contains implementations for the numerical simulation of Gaussian Random fields(RFs). There are already a few Python packages that implement algorithms for simulating Gaussian RFs. However, to the authors' knowledge there doesn't seem to be (an easily found) Python package that contains both exact and approximate methods. So far GaussRF implements one exact (The Circulant Embedding method) and one approximate (Karhunen-Loeve truncation) method. More convincingly, one of the authors has a particular interest in oceanography, and studies the numerical simulation of random fields cognizant of their applications to oceanographic contexts.

### Bibliography
[1] Betz, W., Papaioannou, I., & Straub, D.(2014). Numerical methods for the discretization of random fields by means of the Karhunen-loève expansion. *Computer Methods in Applied Mechanics and Engineering,271,* 109-129.

[2] Dietrich, C. R., & Newsam, G. N. (1993). A fast and exact method for multidimensional Gaussian stochastic simulations. *Water Resources Research, 29(8),* 2861-2869.

[3] Chan, G., & Wood, A. T. (1999). Simulation of stationary Gaussian vector fields. *Statistics and computing, 9(4),* 265-268.

[4] Atkinson, K. E. (1967). The numerical solution of Fredholm integral equations of the second kind. *Siam Journal on Numerical Analysis, 4(3),* 337-348.

[5] Phoon, K. K., Huang, S. P., & Quek, S. T. (2002). Implementation of Karhunen-Loeve expansion for simulation using a wavelet-Galerkin scheme. *Probabilistic Engineering Mechanics, 17(3),* 293-303.
