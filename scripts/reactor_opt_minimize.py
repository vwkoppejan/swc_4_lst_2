# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:14:24 2021

Find the maximum output of a CSTR with a series reaction
A --> B --> C where B is the desired product

@author: ceesharinga
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
def scafun(x,k1,k2,V,C0):
    
    # reaction/flow coefficients
    A = np.array([[-k1*V-x,0,0],[+k1*V, -x-k2*V, 0],[0,k2*V,-x]])
    
    # inflow of A
    b = np.array([-x*C0,0,0])
    
    y = np.linalg.solve(A,b)
    # return the negative of concentration B to maximize
    return -y[1]
    
# parameters
k1 = 0.1             # rate constant in 1/s
k2 = 0.01            # rate constant in 1/s
F =  0.025            # flowrate in m3/s
V = 1.              # volume in m3
C0 = 1.             # initial concentration of A in mol/m3
res = minimize_scalar(scafun,method = 'bounded',bounds=(0.01,0.1),args=(k1,k2,V,C0))   # find the flowrate for the best concentration of B

print(res.x)

