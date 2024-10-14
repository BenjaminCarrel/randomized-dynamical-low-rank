"""
Author: Benjamin Carrel, University of Geneva, 2022
path: matrix_ode_toolbox/utils/spacetime.py
File for spacetime discretization
"""

#%% Imports
import numpy as np
import scipy.sparse as sparse

#%% Centered finite difference matrix O(dx^2)

def centered_1d_dx2(n, dx, periodic=False) -> sparse.spmatrix:
    """
    Discrete centered derivative matrix in 1D (error O(dx^2))
    """
    D = sparse.diags([-1, 1], [-1, 1], shape=(n, n), format='csc') / (2 * dx)
    if periodic:
        D[0, -1] = -1 / (2 * dx)
        D[-1, 0] = 1 / (2 * dx)
    return D

#%% Centered finite difference matrix O(dx^4)

def centered_1d_dx4(n, dx, periodic=False) -> sparse.spmatrix:
    """
    Discrete centered derivative matrix in 1D (error O(dx^4))
    """
    D = sparse.diags([1, -8, 8, -1], [-2, -1, 1, 2], shape=(n, n), format='csc') / (12 * dx)
    if periodic:
        D[0, -2] = 1 / (12 * dx)
        D[0, -1] = -8 / (12 * dx)
        D[1, -1] = 1 / (12 * dx)
        D[-1, 0] = 8 / (12 * dx)
        D[-1, 1] = -1 / (12 * dx)
        D[-2, 0] = -1 / (12 * dx)
    return D


    