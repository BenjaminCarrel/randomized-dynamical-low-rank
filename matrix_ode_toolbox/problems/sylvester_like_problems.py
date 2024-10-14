"""
File for toy problems with a Sylvester-like structure.

Author: Benjamin Carrel, University of Geneva, 2022
"""


#%% IMPORTATIONS
import numpy as np
from scipy import sparse
from matrix_ode_toolbox import SylvesterLikeOde
from low_rank_toolbox import LowRankMatrix
from matrix_ode_toolbox.utils import laplacian_1d_dx2


# %% Allen-Cahn equation
def make_allen_cahn(size: int):
    """
    Allen-Cahn equation
        X' = AX + XA + X - X^3
        X(0) = X0
    where A is the 1D Laplacian (times epsilon) as stencil 1/dx^2 [1 -2 1] in csc format, periodic BC

    Reference: Rodgers and Venturi, 2022, Implicit step-truncation integration of nonlinear PDEs on low-rank tensor manifolds.
    """
    ## PARAMETERS
    epsilon = 0.01

    ## DISCRETIZATION
    xs = np.linspace(0, 2*np.pi, size)
    dx = xs[1] - xs[0]

    ## OPERATOR: Laplacian
    DD = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(size, size), format='csc') / (dx ** 2)
    DD[0, -1] = 1 / (dx ** 2)
    DD[-1, 0] = 1 / (dx ** 2)
    A = epsilon * DD

    ## SOURCE: G(t, X) = X - X^3 (hadamard product)
    def G(t, X):
        if isinstance(X, LowRankMatrix):
            return X - X.hadamard(X.hadamard(X))
        else:
            return X - X**3

    ## DEFINE THE ODE
    ode = SylvesterLikeOde(A, A, G)

    ## INITIAL VALUE
    u = lambda x, y: (np.exp(-np.tan(x)**2) + np.exp(-np.tan(x)**2)) * np.sin(x) * np.sin(y) / (1 + np.exp(np.abs(1/np.sin(-x/2))) + np.exp(np.abs(1/np.sin(-y/2))))
    f = lambda x, y: u(x,y) # - u(x, 2*y) + u(3*x + np.pi, 3*y + np.pi) - 2*u(4*x, 4*y) + 2 * u(5*x, 5*y)
    # NOTE: you can play with it to get different initial rank
    X0 = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            X0[i,j] = u(xs[i], xs[-j])

    return ode, X0
