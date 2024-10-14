"""
Author: Benjamin Carrel, University of Geneva, 2024

File for the dynamical generalized Nystroem methods
"""

#%% Importations
from low_rank_toolbox import LowRankMatrix, SVD, QuasiSVD
import numpy as np
import scipy.linalg as la
from matrix_ode_toolbox.dlra import DlraSolver, AdaptiveDlraSolver
from matrix_ode_toolbox import MatrixOde
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra.randomized_methods import dynamical_rangefinder, dynamical_corangefinder, adaptive_dynamical_corangefinder, adaptive_dynamical_rangefinder

#%% Dynamical generalized Nystroem
class DynamicalGeneralizedNystroem(DlraSolver):
    """
    Class for the dynamical generalized Nystroem method.
    """

    name = 'Dynamical generalized Nystroem'

    def __init__(self, 
                 matrix_ode: MatrixOde, 
                 nb_substeps: int = 1,
                 substep_kwargs: dict = {'solver': 'scipy'},
                 target_rank: int = 10,
                 oversampling_parameters: tuple = (5, 10),
                 power_iterations: tuple = (1, 1),
                 do_ortho_sketching: bool = False,
                 do_augmented: bool = True,
                 seed: int = 1234,
                 **extra_args) -> None:
        super().__init__(matrix_ode, nb_substeps)
        self.substep_kwargs = substep_kwargs
        self.target_rank = target_rank
        self.oversampling_parameters = oversampling_parameters
        self.power_iterations = power_iterations
        self.do_ortho_sketching = do_ortho_sketching
        self.do_augmented = do_augmented
        self.seed = seed
        self.extra_args = extra_args

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Dynamical generalized Nystroem \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f"-- {self.substep_kwargs['solver']} as substep solver \n"
        info += f"-- Target rank: {self.target_rank} \n"
        info += f"-- Oversampling parameters: (p1, p2) = {self.oversampling_parameters} \n"
        info += f"-- Power iterations: (q1, q2) = {self.power_iterations} \n"
        info += f"-- Orthonormal sketching: {self.do_ortho_sketching} \n"
        info += f"-- Augmented basis: {self.do_augmented}"
        return info
    
    def stepper(self, t_subspan: tuple, Y0: LowRankMatrix) -> QuasiSVD:
        "Dynamical generalized Nystroem."
        # CHECK INPUTS
        assert isinstance(Y0, LowRankMatrix), 'X0 must be a LowRankMatrix object.'
        assert len(t_subspan) == 2, 't_subspan must be a tuple of length 2.'
        p1, p2 = self.oversampling_parameters
        q1, q2 = self.power_iterations

        r = self.target_rank
        if Y0.rank < self.target_rank:
            print(f"Warning: rank of initial value is {Y0.rank} while targeted rank is {r}.")
        problem = self.matrix_ode

        # STEP 1: FIND THE RANGE AND CO-RANGE OF THE ODE
        Q1 = dynamical_rangefinder(problem, t_subspan, Y0, self.substep_kwargs, r, p1, q1, self.do_ortho_sketching, seed=self.seed)
        Q2 = dynamical_corangefinder(problem, t_subspan, Y0, self.substep_kwargs, r, p2, q2, self.do_ortho_sketching, seed=self.seed)

        # STEP 1-2: AUGMENT THE BASIS IF NEEDED
        if self.do_augmented:
            Q1 = la.orth(np.column_stack([Q1, Y0.U]))
            Q2 = la.orth(np.column_stack([Q2, Y0.V]))

        # STEP 2: SOLVE THREE SMALL ODEs (can be done in parallel)
        # INITIAL VALUES
        B0 = Y0.dot(Q2, dense_output=True)
        C0 = Y0.T.conj().dot(Q1, dense_output=True)
        D0 = Q1.T.conj().dot(B0)

        # STEP 2-1: COMPUTE B(h)

        problem.select_ode('K', mats_uv=(Q2,))
        B1 = solve_matrix_ivp(problem, t_subspan, B0, **self.substep_kwargs)

        # STEP 2-2: COMPUTE C(h)
        problem.select_ode('L', mats_uv=(Q1,))
        C1 = solve_matrix_ivp(problem, t_subspan, C0, **self.substep_kwargs)

        # STEP 2-3: COMPUTE D(h)
        problem.select_ode('S', mats_uv=(Q1, Q2))
        D1 = solve_matrix_ivp(problem, t_subspan, D0, **self.substep_kwargs)

        # STEP 3: TRUNCATE AND ASSEMBLE THE SOLUTION 
        D1 = SVD.truncated_svd(D1, r=self.target_rank, rtol=1e-12)
        U = B1.dot(D1.V)
        S = np.linalg.inv(D1.S)
        V = C1.dot(D1.U)
        return QuasiSVD(U, S, V).to_svd()

    

#%% Adaptive dynamical generalized Nystroem
class AdaptiveDynamicalGeneralizedNystroem(AdaptiveDlraSolver):
    """
    Class for the adaptive dynamical generalized Nystroem method.

    The method is based on the dynamical generalized Nystroem method, with an adaptive rangefinder.

    The tolerance and probability of failure are used to adapt the number of over-sampling in the rangefinder.

    The rtol and atol are used for the SVD truncation.
    """

    name = 'Adaptive dynamical generalized Nystroem'

    def __init__(self, 
                 matrix_ode: MatrixOde, 
                 nb_substeps: int = 1,
                 rtol: float = 1e-8,
                 atol: float = 1e-12,
                 substep_kwargs: dict = {'solver': 'scipy'},
                 rangefinder_tol: float = 1e-8,
                 failure_probability: float = 1e-6,
                 power_iterations: tuple = (0, 0),
                 do_ortho_sketching: bool = False,
                 do_augmented: bool = True,
                 seed: int = 1234,
                 **extra_args) -> None:
        super().__init__(matrix_ode, nb_substeps, rtol, atol)
        self.substep_kwargs = substep_kwargs
        self.rangefinder_tol = rangefinder_tol
        self.failure_probability = failure_probability
        self.nb_power_iterations = power_iterations
        self.do_ortho_sketching = do_ortho_sketching
        self.do_augmented = do_augmented
        self.seed = seed
        self.extra_args = extra_args

        if power_iterations != (0,0):
            print("Warning: power iterations are not supported yet for the adaptive method.")
            self.nb_power_iterations = (0,0)

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Adaptive dynamical generalized Nystroem \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f"-- Relative tolerance, absolute tolerance: {self.rtol}, {self.atol} \n"
        info += f"-- {self.substep_kwargs['solver']} as substep solver \n"
        info += f"-- Tolerance: {self.rangefinder_tol} \n"
        info += f"-- Failure probability: {self.failure_probability} \n"
        info += f"-- Power iterations: {self.nb_power_iterations} \n"
        info += f"-- Orthonormal sketch: {self.do_ortho_sketching} \n"
        info += f"-- Augmented basis: {self.do_augmented}"
        return info
    
    #%% Dynamical generalized Nystroem (old version - more like double sided DRSVD)
    def stepper(self, t_subspan: tuple, Y0: QuasiSVD) -> QuasiSVD:
        "Adaptive dynamical generalized Nystroem."
        # CHECK INPUTS
        assert len(t_subspan) == 2, 't_subspan must be a tuple of length 2.'
        tol = self.rangefinder_tol
        q1, q2 = self.nb_power_iterations

        problem: MatrixOde = self.matrix_ode

        # STEP 1-1: FIND THE RANGE OF THE ODE
        Q1 = adaptive_dynamical_rangefinder(problem, t_subspan, Y0, self.substep_kwargs, tol, self.failure_probability, q1, self.do_ortho_sketching, seed=self.seed)
        # STEP 1-2: FIND THE CO-RANGE OF THE ODE
        Q2 = adaptive_dynamical_corangefinder(problem, t_subspan, Y0, self.substep_kwargs, tol, self.failure_probability, q2, self.do_ortho_sketching, seed=self.seed)

        # STEP 1-3: AUGMENT THE BASIS IF NEEDED
        if self.do_augmented:
            Q1 = la.orth(np.column_stack([Q1, Y0.U]))
            Q2 = la.orth(np.column_stack([Q2, Y0.V]))

        # Now compute Nystroem approximation
        # STEP 2-1: COMPUTE B(h)
        B0 = Y0.dot(Q2, dense_output=True)
        problem.select_ode('K', mats_uv=(Q2,))
        B1 = solve_matrix_ivp(problem, t_subspan, B0, **self.substep_kwargs)

        # STEP 2-2: COMPUTE C(h)
        C0 = Y0.T.conj().dot(Q1, dense_output=True)
        problem.select_ode('L', mats_uv=(Q1,))
        C1 = solve_matrix_ivp(problem, t_subspan, C0, **self.substep_kwargs)

        # STEP 2-3: COMPUTE D(h)
        D0 = Q1.T.conj().dot(B0)
        problem.select_ode('S', mats_uv=(Q1, Q2))
        D1 = solve_matrix_ivp(problem, t_subspan, D0, **self.substep_kwargs)

        # TRUNCATE AND ASSEMBLE THE SOLUTION
        D1 = SVD.truncated_svd(D1, rtol=self.rtol, atol=self.atol)
        U = B1.dot(D1.V)
        S = np.linalg.inv(D1.S)
        V = C1.dot(D1.U)
        return QuasiSVD(U, S, V).to_svd()
