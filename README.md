
# Guassian Randon Fields 

*A python package for numerical simulation of Gaussian random fields.*

## Description
GaussRF contains implementations for the numerical simulation of Gaussian Random fields(RFs). There are already a few Python packages that implement algorithms for simulating Gaussian RFs. However, to the author's doesn't seem to be (an easily found) Python package that contains both exact and approximate methods. So far GaussRF implements one exact (The Circulant Embedding method) and one approximate (Karhunen-Loeve truncation) method. More convincingly, one of the authors has a particular interest in oceanography, and studies the numerical simulation of random field with an interest in applying this knowledge to oceanographic contexts.


This is a prototype repository to display prototype code for simulating Gaussian Random fields in one- and two-dimensions, as well as a code giving an introduction to the (elements of) the theory. 

From here I plan to touch up the code and store it in a proper Python package. Methods of simulation include 

1. approximate simulations using the Karhunen-Loeve expansion.
2. exact, fast simulations using the Circulant Embedding method. 
_________________________

## Docs 
The Documentation for the GaussRF package can be found [here](https://readthedocs.org/).

______________________


Bibliography
------------------
[1] Betz, W., Papaioannou, I., & Straub, D.(2014). Numerical methods for the discretization of random fields by means of the Karhunen-loève expansion. *Computer Methods in Applied Mechanics and Engineering,271,* 109-129.

[2] Dietrich, C. R., & Newsam, G. N. (1993). A fast and exact method for multidimensional Gaussian stochastic simulations. *Water Resources Research, 29(8),* 2861-2869.

[3] Chan, G., & Wood, A. T. (1999). Simulation of stationary Gaussian vector fields. *Statistics and computing, 9(4),* 265-268.

[4] Atkinson, K. E. (1967). The numerical solution of Fredholm integral equations of the second kind. *SIAM Journal on Numerical Analysis, 4(3),* 337-348.

[5]  Phoon, K. K., Huang, S. P., & Quek, S. T. (2002). Implementation of Karhunen-Loeve expansion for simulation using a wavelet-Galerkin scheme. Probabilistic Engineering Mechanics, 17(3), 293-303.
=======
This is a prototype repository for a package for simulating Gaussian Random fields in one- and two-dimensions. Methods of simulation include (i) approximate simulations using the Karhunen-Loeve expansion and (ii) exact, fast simulations using the Circulant Embedding method. 

