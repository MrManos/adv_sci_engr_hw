import numpy as np
import matplotlib.pyplot as plt

def One_D_F_E(N: int , x_l: float, x_r: float, force: function, bound_con: list[float], step_size: float, ) -> float:
    # Creates the uniform grid and connectivity map
    N_e = N - 1 # number of 1 D elements 
    h = (x_r - x_l)/ (N - 1)
    
    x[N] = 0