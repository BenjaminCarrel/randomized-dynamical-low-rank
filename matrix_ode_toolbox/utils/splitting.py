"""SPLITTING METHODS
"""

#%% IMPORTATIONS
import numpy as np
from typing import Union
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix


def LieTrotter(t_span: tuple, 
               initial_value: Union[ndarray, LowRankMatrix], 
               solver1: object, 
               solver2: object) -> Union[ndarray, LowRankMatrix]:
    """Lie-Trotter splitting.
    For an ODE with a sum
    Y' = F(Y) + G(Y),
    the Lie-Trotter splitting computes
    Y1 = phi_F^h \circ phi_G^h (Y0)
    resulting in an order 1 method in h.

    Args:
        t_span (tuple): time interval (0, h)
        initial_value (Union[ndarray, LowRankMatrix]): initial value
        solver1 (object): first solver function with input (t_span, initial_value)
        solver2 (object): second solver function with input (t_span, initial_value)
    """
    Y0 = initial_value
    Y_half = solver1(t_span, Y0)
    Y1 = solver2(t_span, Y_half)
    return Y1

def Strang(t_span: tuple, 
           initial_value: Union[ndarray, LowRankMatrix], 
           solver1: object, 
           solver2: object) -> Union[ndarray, LowRankMatrix]:
    """Strang splitting.
    For an ODE with a sum
    Y' = F(Y) + G(Y),
    the Strang splitting computes
    Y1 = phi_F^{h/2} \circ phi_G^h \circ phi_F^{h/2} (Y0)
    resulting in an order 2 method in h.

    Args:
        t_span (tuple): time interval (0, h)
        initial_value (Union[ndarray, LowRankMatrix]): initial value
        solver1 (object): first solver function with input (t_span, initial_value)
        solver2 (object): second solver function with input (t_span, initial_value)
    """
    t0 = t_span[0]
    t1 = t_span[1]
    h = t1 - t0
    Y0 = initial_value
    Y_one = solver1((t0, t0 + h/2), Y0)
    Y_two = solver2(t_span, Y_one)
    Y1 = solver1((t0 + h/2, t1), Y_two)
    return Y1