"""
Author: Benjamin Carrel, University of Geneva, 2023

Burgers ODE structure. Subclass of MatrixOde.
"""

#%% IMPORTATIONS
from __future__ import annotations
import numpy as np
from scipy.sparse import spmatrix
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix
from .matrix_ode import MatrixOde




class StochasticBurgersOde(MatrixOde):
    """
    Class for the stochastic Burgers equation, subclass of MatrixOde.
    
    The stochastic Burgers equation is given by:
    X' = DD X - X * (D @ X) 
    where D is the one-dimensional first derivative operator, and DD is the one-dimensional second derivative operator (Laplacian).
    The symbol @ denotes the matrix product and * the element-wise (Hadamard) product.
    """

    # ATTRIBUTES
    name = 'Stochastic Burgers'
    DD = MatrixOde.create_parameter_alias(0)
    D = MatrixOde.create_parameter_alias(1)
    is_stiff = True

    # %% INIT FUNCTION
    def __init__(self, DD: spmatrix, D: spmatrix, **kwargs):
        super().__init__(DD, D, **kwargs)

    def ode_F(self, t: float, X: ndarray | LowRankMatrix) -> ndarray | LowRankMatrix:
        if isinstance(X, LowRankMatrix):
            dX = self.low_rank_ode_F(t, X)
        else:
            dX = self.DD.dot(X) - np.multiply(X, self.D.dot(X))
        return dX
    
    def low_rank_ode_F(self, t: float, X: LowRankMatrix) -> LowRankMatrix:
        return X.dot(self.DD, side='opposite') - X.hadamard(X.dot(self.D, side='opposite'))
    
    def non_linear_field(self, t: float, X: ndarray) -> ndarray:
        return - np.multiply(X, self.D.dot(X))
    
    def linear_field(self, t: float, X: ndarray) -> ndarray:
        return self.DD.dot(X)
    
    def stiff_field(self, t: float, X: ndarray) -> ndarray:
        return self.linear_field(t, X)
    
    def non_stiff_field(self, t: float, X: ndarray) -> ndarray:
        return self.non_linear_field(t, X)
    
