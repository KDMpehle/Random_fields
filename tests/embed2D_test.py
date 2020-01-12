""" Test 2-D circulant embedding: (1-x)*exp(-x) type covariance"""
import numpy as np
from simulation import circ_embed2D
import matplotlib.pyplot as plt
#Some tunable parameters
l1 = 50 # scale length in the x-direction
l2 = 15 # scale length in the y-direction
n = 511 # number of points in x
m = 383 # number of points in y
