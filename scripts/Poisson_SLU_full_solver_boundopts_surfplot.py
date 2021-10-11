# -*- coding: utf-8 -*-
"""
Poisson solver, including with and without reaction, with either Neumann or Dirichlet boundary conditions
Created on Fri Aug 27 13:36:35 2021

@author: ceesharinga
"""

## import the necesarry libraries
from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse import linalg as sla
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix


## define functions

def matrix_construct(n, bound_val, diff_coef, dx, bound_type):
    ''' constructs the coefficient matrix A and condition vector b to solve the poisson equation, based on user-provided conditions
    Input: matrix size, boundary conditions, sources (value and source location), diffusion coefficient, domain size
    The matrix is constructed sparse. Additionally, grid vectors for plotting are initialized''' 

    # initialize matrix
    A = dok_matrix((n*n,n*n), dtype=float)   # output matrix A
    b = np.zeros(n*n)                        # output boundary b
    
    co = diff_coeff/(dx*dx)                  # recurring term - required for later dynamic
    
    # the loop below is used to fill the coefficient matrix 
    # the coefficient is dimensionless, which means the problem is solved in mol/m3
    for i in range(n):                       # fill row-per-row
        for j in range(n):                   # fill entries in row
            c = i*n+j                        # define gridcell number 
            A[c,c] = 4*co                       # if cell is on the diagonal, add inflow from 4 neighbours 
            
            if j!= 0:
                A[c,c-1] = -1*co               # outflow of left neighbour  , unless it's a boundary cell. 
            
            if j!= n-1:
               A[c,c+1] = -1*co                # outflow from right neighbour  , unless it's a boundary cell. 
            
            if i!= 0:
                A[c,c-n] = -1*co               # outflow of top neighbour   , unless it's a boundary cell. 
                
            if i!= n-1:
               A[c,c+n] = -1*co                # outflow of bottom neighbour, unless it's a boundary cell. 
            
            if bound_type == 1:             
                # If the boundary type = 1 we use a NEUMANN BOUNDARY
                # if the node is on a boundary, fixed value boundary conditions are set. This means:
                # an extra subtraction from the gridcell opposite of the ghostcell
                # addition of the source term in the b-vector
                # the equations are mol/m3:  bound_val [mol/m^2/s] * dx [m] / m^2 s = mol/m3 = OK
                if j == 0:                     # if on the left boundary... 
                    A[c,c+1] = -2*co
                    b[c] += 2*bound_val[0]/dx
                if j == n-1:                  # if on the right boundary... 
                    A[c,c-1] = -2*co
                    b[c] += 2*bound_val[1]/dx         
                if i == 0:                    # if on the top boundary...
                    A[c,c+n] = -2*co
                    b[c] += 2*bound_val[2]/dx
                if i == n-1:                  # if on the bottom boundary...
                    A[c,c-n] = -2*co
                    b[c] += 2*bound_val[3]/dx
            else:
                # Otherwise a DIRICHLET boundary is used 
                # if the node is on a boundary, fixed value boundary conditions are set. This means:
                # all non-diagonal entries in that row are 0 
                # the diagonal entry in that row is 1 (1 * Cn = FixedBoundaryValue)
                # the value of b for that entry is equal to the boundary value
                if j == 0:                     # if on the left boundary... 
                    #A[c,:] = 0
                    A[c,c+1] = -1*co
                    b[c] = co*bound_val[0]
                elif j == n-1:                  # if on the right boundary... 
                    #A[c,:] = 0
                    A[c,c-1] = -1*co
                    b[c] = co*bound_val[1]                
                elif i == 0:                    # if on the top boundary...
                   # A[c,:] = 0
                    A[c,c+n] = -1*co
                    b[c] = co*bound_val[2]
                elif i == n-1:                  # if on the bottom boundary...
                    #A[c,:] = 0
                    A[c,n] = -1*co
                    b[c] = co*bound_val[3]
                

                
    print("done fill")
    A = csr_matrix(A)  # CSR format needed for sparse-solve
    return(A,b)
    
def add_sources_firstorder(n, A, source_val):
    ''' adds source terms for 1th order reaction to the appropriate locations in coefficent matrix A'''
    
    for c in range(n*n):                       # fill row-per-row
        A[c,c] = A[c,c] - source_val   # source coeff = source val * dx^2 / diff. coeff  
    return(A)



## define process input parameters
# Units are currently assumed for molar concentration solved. 
# if the solver is used for temperature problems, account for units accordingly
grid_n = 100                                  # grid resolution in 1 direction (the domain size is n*n)
domain_size = 1                              # physical size of the domain (1D, m)
bound_type = bool(0)                         # if 1, Neumann, if 0, Dirichlet
source_switch = bool(1)                      # if 1, sources are added. Otherwise, no source for you.
bound_val = np.array([0, 30, 0, 0])          # boundary values: left, right, top, bottom (if Neumann: mol/m2/s. If Dirichlet: mol/m3)
source_value = 0                             # scalar value for source term = Cx * qs = 5e3 (g/m^3) * 5E-7 mol/g/s  (mol/m3/s)
diff_coeff  = 0.1                             # diffusion coefficient (m^2/s)

# Calculate parameters from inputs
dx = domain_size/(grid_n-1)                  # grid spacing [m]

# Cnstruct the coefficient matrix
(coeffmat, boundvec) = matrix_construct(grid_n, bound_val, diff_coeff, dx, bound_type)

# add sources (if specified)
if source_switch:
    (coeffmat) = add_sources_firstorder(grid_n, coeffmat,source_value)

# solve the system with spsolve
XB = sla.spsolve(coeffmat, boundvec)

# for the surface plot we need 2D-representation of the data
xvec = yvec = np.linspace(0,domain_size,grid_n) # set linear range of points

X,Y = np.meshgrid(xvec,yvec)   #create 2D grids for X and Y coordinates of all oints
Z= np.reshape(XB,(grid_n,grid_n)) # reshape the data into a 2D grid too

# make a fancy plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.azim -= 180
ax.plot_surface(X,Y,Z, vmin = 0, vmax = 30, cmap = 'coolwarm')
ax.set_zlim(0, 30)
