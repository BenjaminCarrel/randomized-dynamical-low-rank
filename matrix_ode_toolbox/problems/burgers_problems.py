"""
File for toy problems with a Burgers structure.

Author: Benjamin Carrel, University of Geneva, 2023
"""

#%% IMPORTATIONS
import numpy as np
from matrix_ode_toolbox import StochasticBurgersOde
from matrix_ode_toolbox.integrate import solve_matrix_ivp


#%% Second variant
def make_stochastic_Burgers(nb_spatial_pts: int = 512, nb_random_samples: int = 32, nu: float = 0.01, sigma: float = 0.001, d: int = 4):
    """
    Generate a one-dimensional stochastic Burgers ODE with random samples on the columns.

    Reference
    ---------
    Paper: https://arxiv.org/abs/2408.16591
    GitHub: https://github.com/BabaeeLab/Implicit-TDB-CUR
    (Our initial conditions are different but the structure is the same)


    The matrix differential equation (after discretization) is:
        X'(t) = D_2 X(t) - X(t) * (D_1 @ X(t))
    where
        D_1 is the first derivative operator
        D_2 is the second derivative operator
        X(t) is a matrix function containing the solution of the ODE

    The initial value is given in the paper.
    The spatial domain is [0, 1].
    The time domain is [0, 1].

    Parameters
    ----------
    nb_spatial_pts: int
        The number of spatial points. Default is 512.
    nb_random_samples: int
        The number of random samples. Default is 32.
    nu: float
        The viscosity parameter. Default is 0.01.
    sigma: float
        The noise parameter. Default is 0.001.
    d: int
        Parameter for the initial value. Default is 4.

    Returns
    -------
    ode: BurgersOde
        The Burgers ode structure with the data generated
    X0: Matrix
        The initial value that can be used out of the box
    """
    # Spatial parameters
    nx = nb_spatial_pts
    xs = np.linspace(0, 1, nx)
    dx = xs[1] - xs[0]
    ## First and second derivative matrices
    D1 = (np.eye(nx, k=1) - np.eye(nx, k=-1)) / (2*dx)
    D2 = nu * (np.eye(nx, k=1) - 2*np.eye(nx) + np.eye(nx, k=-1)) / dx**2
    ## Dirichlet boundary conditions
    D1[0], D1[-1] = 0, 0
    D2[0], D2[-1] = 0, 0

    # Make the ODE
    problem = StochasticBurgersOde(D2, D1)


    # Initial condition 
    np.random.seed(2222)
    s = nb_random_samples
    var = 1
    xi = np.random.randn(s, d)
    kernel_space = np.exp(- (xs[:, None] - xs[None, :])**2 / (2 * var ** 2))
    lam, psi = np.linalg.eig(kernel_space)
    lam = lam.real[:d]
    psi = psi.real[:, :d]
    x0 = 0.5 * np.sin(2 * np.pi * xs) * (np.exp(np.cos(2 * np.pi * xs)) - 1.5)
    X0 = x0[:, None] + sigma * sum([np.sqrt(lam[i]) * psi[:, i, None].dot(xi[:, i, None].T) for i in range(d)])

    return problem, X0





