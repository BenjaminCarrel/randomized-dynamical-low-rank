"""
File for general matrix problems.

Author: Benjamin Carrel, University of Geneva
"""

#%% Imports
import numpy as np
from scipy.linalg import expm
from low_rank_toolbox import LowRankMatrix
from matrix_ode_toolbox import MatrixOde
from numpy import ndarray


#%% TOY PROBLEM
# Ceruti, G., & Lubich, C. (2022). An unconventional robust integrator for dynamical low-rank approximation. BIT Numerical Mathematics, 62(1), 23-44.
def make_matrix_toy_problem(n: int = 100, diag: ndarray = None):
    """
    Toy problem with general matrix form.
    Good for inspecting the behavior of an integrator since we have closed form.

    Reference
        Ceruti, G., & Lubich, C. (2022). An unconventional robust integrator for dynamical low-rank approximation. BIT Numerical Mathematics, 62(1), 23-44.

    ODE
        X' = W_1 X + X + X W_2^T
        W_1 and W_2 are skew-symmetric matrices

    Closed form
        X(t) = e^{t W_1} e^t D e^{t W_2}^T
        D = diag(1/2**i) for i=1,...,n

    Parameters
    ----------
    n: int
        Size of the problem (default 100)
    
    Returns
    -------
    problem: MatrixOde
        Problem in MatrixOde form. The ODE is callable via problem.ode_F
    X0: ndarray
        Initial value
    X_exact: callable
        Callable closed form solution
    """

    # Diagonal matrix
    if diag is None:
        D = np.diag(1/2**np.linspace(1, n, num=n))
    else:
        D = np.diag(diag)

    # Two uniform random skew-symmetric matrices
    np.random.seed(1234)
    W1_tilde = np.random.rand(n, n)
    W2_tilde = np.random.rand(n, n)
    W2_tilde = W1_tilde # For having a symmetric problem
    W1 = (W1_tilde - W1_tilde.T)/2
    W2 = (W2_tilde - W2_tilde.T)/2
    
    # Initial value
    X0 = D

    # ODE
    def F(t, X):
        if isinstance(X, LowRankMatrix):
            X = X.todense()
        return W1.dot(X) + X + X.dot(W2.T)
    
    problem = MatrixOde(D, W1, W2)
    problem.ode_F = F

    # Closed form
    def X_exact(t):
        return expm(t*W1).dot(np.exp(t)*D).dot(expm(t*W2).T)
    
    return problem, X0, X_exact

