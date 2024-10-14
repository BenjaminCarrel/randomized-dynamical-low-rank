"""
Author: Benjamin Carrel, University of Geneva, 2022

Augmented unconventional method for the DLRA.
"""

# %% Imports
import numpy as np
from low_rank_toolbox import SVD, QuasiSVD
import scipy.linalg as la
from matrix_ode_toolbox.dlra.dlra_solver import DlraSolver
from matrix_ode_toolbox import MatrixOde
from matrix_ode_toolbox.integrate import solve_matrix_ivp

#%% Class unconventional
class AugmentedUnconventional(DlraSolver):
    """
    Class for the augmented BUG integrator.
    The method is based on the rank-adaptive unconventional method,
    but the truncation is performed to a fixed rank.
    """

    name = 'Augmented BUG'
    
    def __init__(self, 
                matrix_ode: MatrixOde,
                nb_substeps: int = 1,
                substep_kwargs: dict = {'solver': 'automatic', 'nb_substeps': 1},
                **extra_args) -> None:
        super().__init__(matrix_ode, nb_substeps)
        self.substep_kwargs = substep_kwargs
        self.extra_data = extra_args

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Augmented BUG integrator (Ceruti & Lubich, 2021) \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f"-- {self.substep_kwargs['solver']} as substep solver"
        return info

    def stepper(self, t_subspan: tuple, Y0: QuasiSVD) -> SVD:
        "Augmented unconventional stepper."
        # CHECK INPUTS
        assert len(t_subspan) == 2, 't_subspan must be a tuple of length 2.'
        assert isinstance(Y0, QuasiSVD), 'Y0 must be a QuasiSVD (or SVD).'

        # INITIALISATION
        U0, S0, V0 = Y0.U, Y0.S, Y0.Vt.T
        problem: MatrixOde = self.matrix_ode

        # K-STEP
        K0 = U0.dot(S0)
        problem.select_ode('K', mats_uv=(V0,))
        K1 = solve_matrix_ivp(problem, t_subspan, K0, **self.substep_kwargs)
        U1, _ = la.qr(np.column_stack([U0, K1]), mode='economic')
        M = U1.T.dot(U0)

        # L-STEP
        L0 = V0.dot(S0.T)
        problem.select_ode('L', mats_uv=(U0,))
        L1 = solve_matrix_ivp(problem, t_subspan, L0, **self.substep_kwargs)
        V1, _ = la.qr(np.column_stack([V0, L1]), mode='economic')
        N = V1.T.dot(V0)

        # S-STEP
        S0 = M.dot(S0.dot(N.T))
        problem.select_ode('S', mats_uv=(U1, V1))
        S1 = solve_matrix_ivp(problem, t_subspan, S0, **self.substep_kwargs)

        # SOLUTION
        u, s, vt = la.svd(S1, full_matrices=False)
        U1 = U1.dot(u)
        V1 = V1.dot(vt.T)
        Y1 = SVD(U1, s, V1).truncate(Y0.rank)
        return Y1
