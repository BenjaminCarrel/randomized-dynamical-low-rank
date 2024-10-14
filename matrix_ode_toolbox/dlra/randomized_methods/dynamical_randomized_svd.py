"""
Author: Benjamin Carrel, University of Geneva, 2024

File for dynamical randomized SVD methods
"""

#%% Importations
from low_rank_toolbox import SVD, QuasiSVD, LowRankMatrix
import numpy as np
import scipy.linalg as la
from matrix_ode_toolbox import MatrixOde
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import DlraSolver, AdaptiveDlraSolver
from .dynamical_rangefinder import dynamical_rangefinder, adaptive_dynamical_rangefinder

#%% Dynamical randomized SVD
class DynamicalRandomizedSvd(DlraSolver):
    """
    Class for the dynamical randomized SVD method.
    """

    name = 'Dynamical randomized SVD'
    
    def __init__(self, 
                matrix_ode: MatrixOde,
                nb_substeps: int = 1, 
                substep_kwargs: dict = {'solver': 'scipy'},
                target_rank: int = 10,
                oversampling: int = 5,
                power_iterations: int = 1,
                do_ortho_sketching: bool = True,
                do_augmented: bool = True,
                seed: int = 1234,
                **extra_args) -> None:
        super().__init__(matrix_ode, nb_substeps)
        self.target_rank = target_rank
        self.oversampling = oversampling
        self.nb_power_iterations = power_iterations
        self.do_ortho_sketching = do_ortho_sketching
        self.substep_kwargs = substep_kwargs
        self.seed = seed
        self.extra_args = extra_args
        self.do_augmented = do_augmented

        
    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Dynamical randomized SVD \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f"-- {self.substep_kwargs['solver']} as substep solver \n"
        info += f"-- Target rank: {self.target_rank} \n"
        info += f"-- Oversampling: p={self.oversampling} \n"
        info += f"-- Power iterations: q={self.nb_power_iterations} \n"
        info += f"-- Orthonormal sketching: {self.do_ortho_sketching} \n"
        info += f"-- Augmented basis: {self.do_augmented}"
        return info

    def stepper(self, t_subspan: tuple, Y0: LowRankMatrix) -> QuasiSVD:
        "Dynamical randomized SVD method."
        # CHECK INPUTS
        assert len(t_subspan) == 2, 't_subspan must be a tuple of length 2.'
        assert isinstance(Y0, QuasiSVD), 'Y0 must be a QuasiSVD (or SVD).'
        r = self.target_rank
        if Y0.rank < r:
            print(f"Warning: rank of initial value is {Y0.rank} while targeted rank is {r}.")
        problem = self.matrix_ode
        p, q = self.oversampling, self.nb_power_iterations

        # FIND THE RANGE OF THE ODE
        Qh = dynamical_rangefinder(problem, t_subspan, Y0, self.substep_kwargs, r, p, q, self.do_ortho_sketching, seed=self.seed)
        if self.do_augmented:
            Q = la.orth(np.column_stack([Y0.U, Qh]))
        else:
            Q = Qh

        # POST-PROCESSING WITH THE C-STEP
        C0 = Y0.T.conj().dot(Q, dense_output=True)
        problem.select_ode('L', mats_uv=(Q,))
        C1 = solve_matrix_ivp(problem, t_subspan, C0, **self.substep_kwargs)
        Y1 = SVD.truncated_svd(C1.T.conj(), r=r)
        Y1.U = Q.dot(Y1.U)
        return Y1
        


#%% Adaptive dynamical randomized SVD
class AdaptiveDynamicalRandomizedSvd(AdaptiveDlraSolver):
    """
    Class for the adaptive dynamical randomized SVD method.

    The method is based on the dynamical randomized SVD method, with an adaptive rangefinder.

    The tolerance and probability of failure are used to adapt the number of over-sampling in the rangefinder.

    The rtol and atol are used for the singular values truncation.
    """

    name = 'Adaptive dynamical randomized SVD'
    
    def __init__(self, 
                matrix_ode: MatrixOde,
                nb_substeps: int = 1, 
                rtol: float = 1e-8,
                atol: float = 1e-12,
                substep_kwargs: dict = {'solver': 'scipy'},
                rangefinder_tol: float = 1e-8,
                failure_probability: float = 1e-6,
                power_iterations: int = 1,
                do_ortho_sketching: bool = False,
                do_augmented: bool = True,
                seed: int = 1234,
                **extra_args) -> None:
        super().__init__(matrix_ode, nb_substeps, rtol, atol)
        self.rangefinder_tol = rangefinder_tol
        self.failure_probability = failure_probability
        self.nb_power_iterations = power_iterations
        self.do_ortho_sketching = do_ortho_sketching
        self.substep_kwargs = substep_kwargs
        self.seed = seed
        self.extra_args = extra_args
        self.do_augmented = do_augmented

        if power_iterations != 0:
            print("Warning: power iterations are not supported yet for the adaptive method.")
            self.nb_power_iterations = 0

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Adaptive dynamical randomized SVD \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f"-- Relative tolerance, absolute tolerance: {self.rtol}, {self.atol} \n"
        info += f"-- {self.substep_kwargs['solver']} as substep solver \n"
        info += f"-- Adaptive rangefinder tolerance: {self.rangefinder_tol} \n"
        info += f"-- Failure probability: {self.failure_probability} \n"
        info += f"-- Power iterations: {self.nb_power_iterations} \n"
        info += f"-- Orthonormal sketch: {self.do_ortho_sketching} \n"
        info += f"-- Augmented basis: {self.do_augmented}"
        return info

    def stepper(self, t_subspan: tuple, Y0: QuasiSVD) -> QuasiSVD:
        "Adaptive randomized DLRA method."
        # CHECK INPUTS
        assert len(t_subspan) == 2, 't_subspan must be a tuple of length 2.'
        assert isinstance(Y0, QuasiSVD), 'Y0 must be a QuasiSVD (or SVD).'
        problem: MatrixOde = self.matrix_ode
        
        # FIND THE RANGE OF THE ODE
        Qh = adaptive_dynamical_rangefinder(problem, t_subspan, Y0, self.substep_kwargs, self.rangefinder_tol, self.failure_probability, self.nb_power_iterations, self.do_ortho_sketching, seed=self.seed)

        # C-STEP
        if self.do_augmented:
            Q = la.orth(np.column_stack([Y0.U, Qh]))
        else:
            Q = Qh
        C0 = Y0.T.conj().dot(Q, dense_output=True)
        problem.select_ode('L', mats_uv=(Q,))
        C1 = solve_matrix_ivp(problem, t_subspan, C0, **self.substep_kwargs)
        Y1 = SVD.truncated_svd(C1.T.conj(), rtol=self.rtol, atol=self.atol)

        # SOLUTION
        Y1.U = Q.dot(Y1.U)
        return Y1
