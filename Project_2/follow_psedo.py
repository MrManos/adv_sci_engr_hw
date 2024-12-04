import numpy as np
import matplotlib.pyplot as plt

def One_D_F_E(L:int, N: int , x_l: float, x_r: float, force: function, bound_con: np.array[float], step_size: float, ) -> float:
    # Creates the space array 
    x = np.linspace(0,L,N)
    # Creates the uniform grid and connectivity map
    N_e = N - 1 # number of 1 D elements 
    h = (x_r - x_l)/ (N - 1)
    
    # Connectivity map
    connectivity = np.zeros((N_e, 2), dtype=int)
    for i in range(N_e):
        connectivity[i, 0] = i       # Left node
        connectivity[i, 1] = i + 1  # Right node
    
    
    k[N][N] = 0
    f[N] = 0
    klocal[2][2] = 0
    flocal[2] = 0
    
    
    # Gaussian quadrature points and weights for 2nd order
    quad_points = [-1/np.sqrt(3), 1/np.sqrt(3)]
    quad_weights = [1, 1]
    

    for j in range(N_e):
        for l to 2:
            
    
    
# define parent grid -1 <= \xi <= 1 basis functions and derivatives
def phi_1(xi):
        return (1 - xi) / 2  # Basis function 1
    
def phi_2(xi):
    return (1 + xi) / 2  # Basis function 2
    
def dphi_1_dxi(xi):
    return -0.5  # Derivative of phi_1
    
def dphi_2_dxi(xi):
    return 0.5  # Derivative of phi_2

