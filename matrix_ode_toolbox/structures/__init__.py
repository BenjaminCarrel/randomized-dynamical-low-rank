"""
Author: Benjamin Carrel, University of Geneva, 2022

The structures module contains the classes used to represent the different matrix ODEs.
"""
from .matrix_ode import MatrixOde
from .sylvester_ode import SylvesterOde
from .sylvester_like_ode import SylvesterLikeOde
from .burgers_ode import StochasticBurgersOde
from .vlasov_poisson_ode import VlasovPoissonOde