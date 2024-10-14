"""
Author: Benjamin Carrel, University of Geneva, 2023

Sylvester-like ODE structure. Subclass of MatrixOde.
"""

# %% IMPORTATIONS
from __future__ import annotations
from scipy.sparse import spmatrix
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix
from .matrix_ode import MatrixOde
from typing import Callable

Matrix = ndarray | spmatrix | LowRankMatrix

# %% CLASS SYLVESTER-LIKE
class SylvesterLikeOde(MatrixOde):
    """
    Class for Sylvester-like equations. Subclass of MatrixOde.

    Sylvester-like differential equation : 
    X'(t) = A X(t) + X(t) B + G(t, X(t)).
    Initial value given by X(t_0) = X0.

    Typically, A and B are sparse matrices, and G is a non-linear function.

    The linear field is assumed to be stiff, and the non-linear field is assumed to be non-stiff. To change this, edit the stiff_field and non_stiff_field methods.
    """

    #%% ATTRIBUTES
    name = 'Sylvester-like'
    A = MatrixOde.create_parameter_alias(0)
    B = MatrixOde.create_parameter_alias(1)
    G = MatrixOde.create_parameter_alias(2)

    def __init__(self, A: Matrix, B: Matrix, G: Callable, **kwargs):
        """Sylvester-like differential equation: X'(t) = A X(t) + X(t) B + G(X(t))."""
        # Check inputs
        assert isinstance(A, Matrix), "A must be a sparse matrix"
        assert isinstance(B, Matrix), "B must be a sparse matrix"
        assert callable(G), "G must be a function"

        # INITIALIZATION
        super().__init__(A, B, G, **kwargs)

    @property
    def shape(self) -> tuple:
        return (self.A.shape[0], self.B.shape[1])

    def ode_F(self, t: float, X: Matrix) -> Matrix:
        """Return the right-hand side of the ODE."""
        if isinstance(X, LowRankMatrix):
            return X.dot(self.A, side='opposite') + X.dot(self.B) + self.G(t, X)
        else:
            return self.G(t, X) + self.A.dot(X) + self.B.T.dot(X.T).T
    
    def preprocess_ode(self):
        "Preprocess the ODE -> compute the factors of the selected ODE"
        super().preprocess_ode()
        A, B, G = self.A, self.B, self.G
        ode_type, mats_uv = self.ode_type, self.mats_uv
        if ode_type == "F":
            self.Ar, self.Br, self.Gr = A, B, G
            return self
        elif ode_type == "K":
            (V,) = mats_uv
            self.Ar = A
            self.Br = V.T.conj().dot(B.dot(V))
            self.Gr = lambda t, K: G(t, K.dot(V.T.conj())).dot(V)
        elif ode_type == "L": # don't forget: A and B are switched due to the transpose
            (U,) = mats_uv
            self.Ar = B.T.conj()
            self.Br = U.T.conj().dot(A.T.conj().dot(U))
            self.Gr = lambda t, L: G(t, U.dot(L.T.conj())).T.conj().dot(U)
        elif ode_type == "S" or ode_type == "minus_S":
            (U, V) = mats_uv
            self.Ar = U.T.conj().dot(A.dot(U))
            self.Br = V.T.conj().dot(B.dot(V))
            self.Gr = lambda t, S: U.T.conj().dot(G(t, U.dot(S.dot(V.T.conj()))).dot(V))
        elif ode_type == "B": # K-like 
            (Omega, Wh) = mats_uv
            self.Ar = A
            self.Br = Wh.dot(B.dot(Omega))
            self.Gr = lambda t, B: G(t, B.dot(Wh)).dot(Omega)
        elif ode_type == "C": # L-like
            (Omega, Zh) = mats_uv
            self.Ar = B.T.conj()
            self.Br = Zh.dot(A.T.conj().dot(Omega))
            self.Gr = lambda t, C: G(t, Zh.T.conj().dot(C.T.conj())).T.conj().dot(Omega)


    def ode_K(self, t: float, K: ndarray) -> ndarray:
        "Return the right-hand side of the K-ODE."
        return self.Ar.dot(K) + K.dot(self.Br) + self.Gr(t, K)
    
    def ode_L(self, t: float, L: ndarray) -> ndarray:
        "Return the right-hand side of the L-ODE."
        return self.Ar.dot(L) + L.dot(self.Br) + self.Gr(t, L)
        
    def ode_S(self, t: float, S: ndarray) -> ndarray:
        "Return the right-hand side of the S-ODE."
        return self.Ar.dot(S) + S.dot(self.Br) + self.Gr(t, S)
    
    def linear_field(self, t: float, X: Matrix) -> Matrix:
        """Return the linear field of the ODE."""
        if isinstance(X, LowRankMatrix):
            return X.dot(self.Ar, side='left') + X.dot(self.Br, side='right')
        else:
            return self.Ar.dot(X) + self.Br.T.dot(X.T).T

    def non_linear_field(self, t: float, X: Matrix, **extra_args) -> Matrix:
        return self.Gr(t, X, **extra_args)

    def stiff_field(self, t: float, X: Matrix, **extra_args) -> Matrix:
        return self.linear_field(t, X, **extra_args)

    def non_stiff_field(self, t: float, Y: Matrix, **extra_args) -> Matrix:
        return self.non_linear_field(t, Y, **extra_args)

    

    