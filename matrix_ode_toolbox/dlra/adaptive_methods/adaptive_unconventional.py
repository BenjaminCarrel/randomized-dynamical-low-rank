"""
Author: Benjamin Carrel, University of Geneva, 2022

Rank-adaptive unconventional method for the DLRA.
See Ceruti and Lubich, 2021.
"""

# %% Imports
import numpy as np
from low_rank_toolbox import SVD, QuasiSVD
import scipy.linalg as la
from matrix_ode_toolbox.dlra.adaptive_dlra_solver import AdaptiveDlraSolver
from matrix_ode_toolbox import MatrixOde
from matrix_ode_toolbox.integrate import MatrixOdeSolver, solve_matrix_ivp

#%% Class unconventional
class AdaptiveUnconventional(AdaptiveDlraSolver):
    """
    Class for the rank-adaptive unconventional DLRA method.
    See Ceruti and Lubich, 2021.
    """

    name = 'Rank-adaptive unconventional'
    
    def __init__(self, 
                matrix_ode: MatrixOde,
                nb_substeps: int = 1, 
                rtol: float = 1e-8,
                atol: float = 1e-8,
                substep_kwargs: str | MatrixOdeSolver = {'solver': 'automatic', 'nb_substeps': 1}, 
                **extra_args) -> None:
        super().__init__(matrix_ode, nb_substeps, rtol, atol)
        self.substep_kwargs = substep_kwargs
        self.extra_args = extra_args

    @property
    def info(self) -> str:
        """Return the info string."""
        info = f'Rank-adaptive unconventional (Ceruti & Lubich 2021) \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f"-- {self.substep_kwargs['solver']} as substep solver \n"
        info += f'-- Relative tolerance = {self.rtol} \n'
        info += f'-- Absolute tolerance = {self.atol}'
        return info


    def stepper(self, t_subspan: tuple, Y0: QuasiSVD) -> SVD:
        """
        Unconventional DLRA method.
        """
        # CHECK INPUTS
        assert len(t_subspan) == 2, 't_subspan must be a tuple of length 2.'
        assert isinstance(Y0, QuasiSVD), 'Y0 must be a QuasiSVD (or SVD).'

        # INITIALISATION
        U0, S0, V0 = Y0.U, Y0.S, Y0.Vt.T
        problem: MatrixOde = self.matrix_ode

        # K-STEP
        K0 = U0.dot(S0)
        problem.select_ode('K', (V0,))
        K1 = solve_matrix_ivp(problem, t_subspan, K0, **self.substep_kwargs)
        U1, _ = la.qr(np.column_stack([U0, K1]), mode='economic')
        M = U1.T.dot(U0)

        # L-STEP
        L0 = V0.dot(S0.T)
        problem.select_ode('L', (U0,))
        L1 = solve_matrix_ivp(problem, t_subspan, L0, **self.substep_kwargs)
        V1, _ = la.qr(np.column_stack([V0, L1]), mode='economic')
        N = V1.T.dot(V0)

        # S-STEP
        S0 = M.dot(S0.dot(N.T))
        problem.select_ode('S', (U1, V1))
        S1 = solve_matrix_ivp(problem, t_subspan, S0, **self.substep_kwargs)

        # SOLUTION
        u, s, vt = la.svd(S1, full_matrices=False)
        U1 = U1.dot(u)
        V1 = V1.dot(vt.T)
        Y1 = SVD(U1, s, V1).truncate(rtol=self.rtol, atol=self.atol)
        return Y1




