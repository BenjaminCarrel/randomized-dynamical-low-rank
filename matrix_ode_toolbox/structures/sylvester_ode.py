"""
Author: Benjamin Carrel, University of Geneva, 2022

Sylvester ODE structure. Subclass of MatrixOde.
"""


# %% IMPORTATIONS
from __future__ import annotations
from matrix_ode_toolbox.structures.matrix_ode import Matrix
from scipy.sparse import spmatrix
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix
from .matrix_ode import MatrixOde

# %% CLASS SYLVESTER
class SylvesterOde(MatrixOde):
    """
    Subclass of MatrixOde. Specific to the Sylvester equation.

    Sylvester differential equation : X'(t) = A X(t) + X(t) B + C.
    Initial value given by X(t_0) = X0.
    
    Typically, A and B are sparse matrices, and C is a low-rank matrix.
    """

    #%% ATTRIBUTES
    name = 'Sylvester'
    A = MatrixOde.create_parameter_alias(0)
    B = MatrixOde.create_parameter_alias(1)
    C = MatrixOde.create_parameter_alias(2)

    #%% FUNDAMENTALS
    def __init__(self, A: ndarray | spmatrix, B: ndarray | spmatrix, C: LowRankMatrix, **kwargs):
        """Sylvester differential equation: X'(t) = A X(t) + X(t) B + C."""
        # Check inputs
        assert isinstance(A, ndarray | spmatrix), "A must be a dense or sparse matrix"
        assert isinstance(B, ndarray | spmatrix), "B must be a dense or sparse matrix"
        assert isinstance(C, LowRankMatrix), "C must be a LowRankMatrix"

        # INITIALIZATION
        super().__init__(A, B, C, **kwargs)

    @property
    def shape(self) -> tuple:
        "Shape of the Sylvester equation"
        return self.C.shape

    def preprocess_ode(self):
        "Preprocess the ODE -> compute the factors of the selected ODE"
        super().preprocess_ode()
        A, B, C = self.A, self.B, self.C
        ode_type, mats_uv = self.ode_type, self.mats_uv
        if ode_type == "F":
            self.Ar, self.Br, self.Cr = A, B, C
            return self
        elif ode_type == "K":
            (V,) = mats_uv
            self.Ar = A
            self.Br = V.T.conj().dot(B.dot(V))
            self.Cr = C.dot(V).todense()
        elif ode_type == "L": # don't forget: A and B are switched due to the transpose
            (U,) = mats_uv
            self.Ar = B.T.conj()
            self.Br = U.T.conj().dot(A.dot(U))
            self.Cr = C.dot(U).todense()
        elif ode_type == "S" or ode_type == "minus_S":
            (U, V) = mats_uv
            self.Ar = U.T.conj().dot(A.dot(U))
            self.Br = V.T.conj().dot(B.dot(V))
            self.Cr = C.dot(V).dot(U.T.conj(), side='opposite').todense()
        elif ode_type == "B":
            (Omega, Wh) = mats_uv
            self.Ar = A
            self.Br = Wh.dot(B.dot(Omega))
            self.Cr = C.dot(Omega, dense_output=True)
        elif ode_type == "C":
            (Omega, Zh) = mats_uv
            self.Ar = B.T.conj()
            self.Br = Zh.dot(A.T.conj().dot(Omega))
            self.Cr = C.T.conj().dot(Omega, dense_output=True)
        elif ode_type == "D":
            (Y, Zh, X, Wh) = mats_uv
            self.Ar = Y.T.conj().dot(A.dot(Zh.T.conj()))
            self.Br = Wh.dot(B.dot(X))
            self.Cr = C.dot(Y, side='opposite').dot(X, dense_output=True)

    #%% VECTOR FIELDS
    def ode_F(self, t: float, X: ndarray | LowRankMatrix) -> ndarray:
        "F(X) = A X + X B + C"
        if isinstance(X, LowRankMatrix):
            dY = X.dot(self.A, side='left') + X.dot(self.B, side='right') + self.C
        else:
            dY = self.C + self.A.dot(X) + self.B.T.dot(X.T).T
        return dY
    
    def ode_K(self, t: float, K: ndarray) -> ndarray:
        return self.Ar.dot(K) + K.dot(self.Br) + self.Cr
    
    def ode_S(self, t: float, S: ndarray) -> ndarray:
        return self.Ar.dot(S) + S.dot(self.Br) + self.Cr

    def ode_L(self, t: float, L: ndarray) -> ndarray:
        return self.Ar.dot(L) + L.dot(self.Br) + self.Cr
        
    def ode_B(self, t: float, B: ndarray) -> ndarray:
        return self.Ar.dot(B) + B.dot(self.Br) + self.Cr
    
    def ode_C(self, t: float, C: ndarray) -> ndarray:
        return self.Ar.dot(C) + C.dot(self.Br) + self.Cr
        
    def linear_field(self, t: float, X: ndarray | LowRankMatrix) -> ndarray:
        "Linear field of the equation"
        if isinstance(X, LowRankMatrix):
            dY = X.dot(self.Ar, side='left') + X.dot(self.Br, side='right')
        else:
            dY = self.Ar.dot(X) + self.Br.T.dot(X.T).T
        return dY

    def non_linear_field(self, t: float, X: ndarray | LowRankMatrix) -> LowRankMatrix:
        "Non linear field of the equation"
        return self.Cr

    