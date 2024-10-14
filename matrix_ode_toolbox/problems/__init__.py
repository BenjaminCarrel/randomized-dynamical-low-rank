"""
Folder for generating toy problems.
The functions below generate an ODE and an initial value that can be used out of the box.
The problems are consistently generated so that the results are always the same.

Author: Benjamin Carrel, University of Geneva 2023
"""

from .matrix_problems import make_matrix_toy_problem
from .sylvester_problems import make_lyapunov_heat_square_dirichlet_kressner
from .sylvester_like_problems import make_allen_cahn
from .burgers_problems import make_stochastic_Burgers
from .vlasov_poisson_problems import make_landau_damping, make_two_stream
