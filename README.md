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


