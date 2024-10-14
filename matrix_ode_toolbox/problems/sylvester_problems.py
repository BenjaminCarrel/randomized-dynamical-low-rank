"""
File for toy problems with a Sylvester structure.

Author: Benjamin Carrel, University of Geneva, 2023
"""

# %% IMPORTATIONS
import numpy as np
import numpy.linalg as la
import scipy.sparse as sps
from low_rank_toolbox import SVD
from matrix_ode_toolbox import SylvesterOde
from matrix_ode_toolbox.integrate import solve_matrix_ivp


def make_lyapunov_heat_square_dirichlet_kressner(size, skip_iv=False, alpha = 1):
    """
    Generate a Lyapunov ODE that models a 2D heat propagation on the square [0,pi]x[0,pi] with Dirichlet BC.

    The data (initial value and source term) are generated as in the draft of Hysan and Kressner on randomized low-rank Runge-Kutta methods.
    Reference: Randomized low-rank Runge-Kutta methods (Lam and Kressner, 2024).

    Parameters
    ----------
    size: int
        The size of the ODE
    skip_iv: bool
        Whether to skip the instabilities due to the randomness in the initial value
    alpha: float
        The scaling factor for the source term

    Returns
    -------
    ode: LyapunovOde
        The Lyapunov ode structure with the data generated
    X0: Matrix
        The initial value that can be used out of the box
    """
    x_space = np.linspace(-np.pi, np.pi, num=size)
    dx = x_space[1] - x_space[0]

    ## OPERATOR: A is the 1D Laplacian as stencil 1/dx^2 [1 -2 1] in csc format, dirichlet BC
    A = (1/dx)**2 * sps.diags([1, -2, 1], [-1, 0, 1], shape=(size, size), format="csc")

    ## SOURCE: low-rank matrix C
    C = np.zeros((size, size))
    for k in np.arange(1, 21):
        for i in range(size):
            for j in range(size):
                C[i, j] = np.sum([10**(-k+1) * np.exp(-k*(x_space[i]**2 + x_space[j]**2)) for k in range(1, 11)], axis=0)
    C = alpha * C / la.norm(C, 'fro')
    C = SVD.from_dense(C)

    ## DEFINE THE ODE
    ode = SylvesterOde(A, A, C)

    ## INITIAL VALUE: X_0
    def b(k):
        if k == 1:
            return 1
        else:
            return 5 * np.exp(-(7 + 0.5 * (k-2)))
    X0 = np.zeros((size, size))
    for k in np.arange(1, 21):
        for i in range(size):
            for j in range(size):
                X0[i, j] = b(k) * np.sin(k*x_space[i]) * np.sin(k*x_space[j])

    if skip_iv:
        ref_solver = 'exponential_runge_kutta'
        ref_solver_kwargs = {'order': 2, 'nb_substeps': 1}
        X0 = solve_matrix_ivp(ode, (0, 1e-4), X0, solver=ref_solver, solver_kwargs=ref_solver_kwargs)

    return ode, X0