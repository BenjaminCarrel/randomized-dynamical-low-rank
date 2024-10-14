"""
Author: Benjamin Carrel, University of Geneva, 2024

Vlasov-Poisson ODE structure. Subclass of MatrixOde.
"""

#%% IMPORTATIONS
from __future__ import annotations
import numpy as np
from scipy.sparse import spmatrix
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix
from .matrix_ode import MatrixOde
import warnings as warning


class VlasovPoissonOde(MatrixOde):
    """
    Class for the Vlasov-Poisson equation, subclass of MatrixOde.
    
    The Vlasov-Poisson equation is given by:
    A' = - D_x A V - E_t A D_v
    where:
    - A(t) = A(t, x_i, v_j)_{i,j} is the distribution function,
    - D_x is the first derivative operator in the x direction,
    - D_v is the first derivative operator in the v direction,
    - V is the velocity field,
    - E_t is the electric field.
    """

    # Attributes
    name = 'Vlasov-Poisson'
    Dx = MatrixOde.create_parameter_alias(0)
    Dv = MatrixOde.create_parameter_alias(1)
    diagV = MatrixOde.create_parameter_alias(2)
    dx = MatrixOde.create_parameter_alias(3)
    dv = MatrixOde.create_parameter_alias(4)

    # Init function
    def __init__(self, Dx: spmatrix | ndarray, Dv: spmatrix | ndarray, diagV: ndarray, dx: float, dv: float, **kwargs):
        if len(diagV.shape) == 1:
            diagV = np.diag(diagV)
        else:
            # Check if V is diagonal
            diagV_bis = np.diag(np.diag(diagV))
            if not np.allclose(diagV, diagV_bis):
                raise ValueError('diagV must be diagonal.')
        super().__init__(Dx, Dv, diagV, dx, dv, **kwargs)

    def electric_field(self, rho: ndarray) -> ndarray:
        "Compute the electric field."
        dxE = np.ones_like(rho) - rho
        # Integration
        fftdxE= np.fft.fft(dxE).flatten()
        n =fftdxE.size
        freq = np.fft.fftfreq(n, d=self.dx).flatten()
        Einter = np.zeros(n, dtype=complex)
        Einter[1:] = fftdxE[1:]/(1j*freq[1:]*2*np.pi) # Division by zero avoided
        E=np.fft.ifft(Einter)
        return E.real

    def ode_F(self, t: float, A: ndarray | LowRankMatrix) -> ndarray:
        if isinstance(A, LowRankMatrix): # Sanity check
            A = A.todense()
            # print('Warning in VlasovPoissonOde.ode_F: A is a LowRankMatrix. It is converted to a dense matrix.')
        rho = self.dv * np.sum(A, axis=1)
        E = self.electric_field(rho)
        diagE = np.diag(E)
        dA = - self.Dx.dot(A).dot(self.diagV) - diagE.dot(A).dot(self.Dv)
        return dA
    
    def preprocess_ode(self):
        "Preprocess the ODE."
        super().preprocess_ode()
        Dx = self.Dx
        Dv = self.Dv
        diagV = self.diagV
        ode_type, mats_uv = self.ode_type, self.mats_uv
        if ode_type == 'K':
            (V,) = mats_uv
            self.Dvr = V.T.conj().dot(Dv.dot(V))
            self.diagVr = V.T.conj().dot(diagV.dot(V))
            self.sumVt = self.dv * np.sum(V.T.conj(), axis=1)
        elif ode_type == 'L':
            (U,) = mats_uv
            self.Dxr = U.T.conj().dot(Dx.T.conj().dot(U))
        elif ode_type == 'S' or ode_type == 'minus_S':
            (U, V) = mats_uv
            self.Dxr = U.T.conj().dot(Dx.dot(U))
            self.Dvr = V.T.conj().dot(Dv.dot(V))
            self.diagVr = V.T.conj().dot(diagV.dot(V))
            self.sumVt = self.dv * np.sum(V.T.conj(), axis=1)
        elif ode_type == 'B':
            (Omega, Wh) = mats_uv
            self.sumWh = self.dv * np.sum(Wh, axis=1)
            self.Dvr = Wh.dot(Dv.dot(Omega))
            self.diagVr = Wh.dot(diagV.dot(Omega))
        elif ode_type == 'C':
            (Omega, Zh) = mats_uv
            self.Dxr = Zh.dot(Dx.T.conj().dot(Omega))
        elif ode_type == 'D':
            (Y, Zh, X, Wh) = mats_uv
            self.Dxr = Y.T.conj().dot(Dx.dot(Zh.T.conj()))
            self.Dvr = Wh.dot(Dv.dot(X))
            self.diagVr = Wh.dot(diagV.dot(X))
            self.sumWh = self.dv * np.sum(Wh, axis=1)
            
    def ode_K(self, t: float, K: ndarray) -> ndarray:
        rho = K.dot(self.sumVt)
        E = self.electric_field(rho)
        diagE = np.diag(E)
        dK = - self.Dx.dot(K.dot(self.diagVr)) - diagE.dot(K.dot(self.Dvr))
        return dK
    
    def ode_L(self, t: float, L: ndarray) -> ndarray:
        (U,) = self.mats_uv
        rho = self.dv * U.dot(np.sum(L.T.conj(), axis=1))
        E = self.electric_field(rho)
        diagE = U.T.conj().dot(np.diag(E).T.conj().dot(U))
        dL = - self.diagV.T.conj().dot(L.dot(self.Dxr)) - self.Dv.T.conj().dot(L.dot(diagE))
        return dL
    
    def ode_S(self, t: float, S: ndarray) -> ndarray:
        (U, V) = self.mats_uv
        rho = U.dot(S.dot(self.sumVt))
        E = self.electric_field(rho)
        diagE = U.T.conj().dot(np.diag(E).dot(U))
        dS = - self.Dxr.dot(S.dot(self.diagVr)) - diagE.dot(S.dot(self.Dvr))
        return dS
    
    def ode_B(self, t: float, B: ndarray) -> ndarray:
        rho = self.dv * B.dot(self.sumWh)
        E = self.electric_field(rho)
        diagE = np.diag(E)
        dB = - self.Dx.dot(B.dot(self.diagVr)) - diagE.dot(B.dot(self.Dvr))
        return dB
    
    def ode_C(self, t: float, C: ndarray) -> ndarray:
        (Omega, Zh) = self.mats_uv
        rho = Zh.T.conj().dot(np.sum(C.T.conj(), axis=1))
        E = self.electric_field(rho)
        diagEr = Zh.dot(np.diag(E).T.conj().dot(Omega))
        dC = - self.diagV.T.conj().dot(C.dot(self.Dxr)) - self.Dv.T.conj().dot(C.dot(diagEr))
        return dC
    
