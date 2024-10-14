"""
Author: Benjamin Carrel, University of Geneva, 2024

File for the dynamical range finder and co-range finder methods
"""

#%% Importations
import numpy as np
import scipy.linalg as la
from matrix_ode_toolbox import MatrixOde
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from low_rank_toolbox import LowRankMatrix, QuasiSVD
import warnings
from typing import Any

#%% The dynamical rangefinder method
def dynamical_rangefinder(matrix_ode: MatrixOde,
                          t_subspan: tuple, 
                          Y0: LowRankMatrix,
                          substep_kwargs: dict = {'solver': 'scipy'},
                          target_rank: int = 0, 
                          oversampling_parameter: int = 10,
                          power_iterations: int = 1,
                          do_ortho_sketch: bool = False,
                          return_dict: bool = False,
                          seed: int = 1234) -> Any:
        """
        The dynamical range finder method.

        Parameters
        ----------
        matrix_ode : MatrixOde
            The matrix ODE with selection of the fields: 'K' and 'L' for orthogonal sketching or 'B' and 'C' for non-orthogonal sketching.
        t_subspan : tuple
            The time span for the integration.
        Y0 : LowRankMatrix
            The initial value as a LowRankMatrix.
        substep_kwargs : dict, optional
            The solver for the substep, by default {'solver': 'scipy'}.
        target_rank : int, optional
            The target rank of the approximation, by default None.
        oversampling_parameter : int, optional
            The oversampling parameter, by default 10.
        power_iterations : int, optional
            The number of power iterations, by default 1.
        do_ortho_sketch : bool, optional
            If the sketch matrix is orthogonal, by default False.
        seed : int, optional
            The seed for the random number generator, by default 1234.

            
        Returns
        -------
        Q: np.ndarray
            A matrix with orthogonal columns approximating the range of the ODE at the final time. Size m x (r+p).
        """

        # CHECK INPUTS
        assert len(t_subspan) == 2, 't_subspan must be a tuple of length 2.'
        assert isinstance(Y0, (np.ndarray, QuasiSVD)), 'Y0 must be a LowRankMatrix.'

        # INITIALISATION
        A0 = Y0
        (m, n) = A0.shape
        problem: MatrixOde = matrix_ode
        if target_rank == 0:
            warnings.warn('The target rank is not provided, so it is set to the rank of the initial value.')
            target_rank = Y0.rank

        # SKETCH RANDOM MATRIX
        np.random.seed(seed)
        Omega = np.random.randn(n, target_rank + oversampling_parameter)
        if do_ortho_sketch:
            Omega = la.orth(Omega)
            is_ortho = True
        else:
            is_ortho = False
        Omega.astype(A0.dtype)
        
        # B-STEP: FIND THE RANGE OF THE ODE
        if is_ortho:
            # Omega = la.orth(np.column_stack([A0.U, Omega])) # Augmented basis
            problem.select_ode('K', mats_uv=(Omega,))
        else:
            Wh = la.solve(Omega.T.conj().dot(Omega), Omega.T.conj())
            problem.select_ode('B', mats_uv=(Omega, Wh))
        B0 = A0.dot(Omega, dense_output=True)
        B1 = solve_matrix_ivp(problem, t_subspan, B0, dense_output=True, **substep_kwargs)
        Q, _, _ = la.qr(B1, mode='economic', pivoting=True)

        # Power iterations
        for _ in range(power_iterations):
            # C-STEP
            C0 = A0.T.conj().dot(Q, dense_output=True)
            problem.select_ode('L', mats_uv=(Q,))
            C1 = solve_matrix_ivp(problem, t_subspan, C0, dense_output=True, **substep_kwargs)
            Q, _, _ = la.qr(C1, mode='economic', pivoting=True)
            # B-STEP
            B0 = A0.dot(Q, dense_output=True)
            problem.select_ode('K', mats_uv=(Q,))
            B1 = solve_matrix_ivp(problem, t_subspan, B0, dense_output=True, **substep_kwargs)
            Q, _, _ = la.qr(B1, mode='economic', pivoting=True)

        return Q

## The adaptive dynamical rangefinder method
def adaptive_dynamical_rangefinder(matrix_ode: MatrixOde,
                                   t_subspan: tuple, 
                                   Y0: LowRankMatrix,
                                   substep_kwargs: dict = {'solver': 'scipy'},
                                   tol: float = 1e-8,
                                   failure_prob: float = 1e-6,
                                   power_iterations: int = 1,
                                   do_ortho_sketch: bool = False,
                                   seed: int = 1234) -> Any:
        """
        The adaptive dynamical range finder method.

        The tolerance corresponds to the error made by the approximation of the range of the ODE at the final time : 
        ||A(h) - Q Q^H A(h)||_F <= tol
        The failure probability is the probability of not satisfying the tolerance.
        Low failure probability implies larger sampling method.
        The error is estimated with the samples used during the method.

        Parameters
        ----------
        matrix_ode : MatrixOde
            The matrix ODE with selection of the fields: 'K' and 'L' for orthogonal sketching or 'B' and 'C' for non-orthogonal sketching.
        t_subspan : tuple
            The time span for the integration.
        Y0 : LowRankMatrix
            The initial value as a LowRankMatrix of size m x n.
        substep_kwargs : dict, optional
            The solver for the substep, by default {'solver': 'scipy'}.
        tol : float, optional
            The tolerance for the adaptive method, by default 1e-8.
        failure_prob : float, optional
            The failure probability for the adaptive method, by default 1e-6.
        power_iterations : int, optional
            The number of power iterations, by default 1.
        do_ortho_sketch : bool, optional
            If the sketch matrix is orthogonal, by default False.
        seed : int, optional
            The seed for the random number generator, by default 1234.
            
        Returns
        -------
        Q: np.ndarray
            A matrix with orthogonal columns approximating the range of the ODE at the final time. Size m x (r+p).
        """

        # CHECK INPUTS
        assert len(t_subspan) == 2, 't_subspan must be a tuple of length 2.'
        assert isinstance(Y0, (np.ndarray, QuasiSVD)), 'initial_value must be an array or a QuasiSVD (or SVD).'

        # INITIALISATION
        A0 = Y0
        (m, n) = A0.shape
        problem: MatrixOde = matrix_ode
        q = power_iterations

        # Compute the sketch size according to the failure probability
        r = int(np.ceil(- np.log(failure_prob) / np.log(10)))
        tol = tol / (10 * np.sqrt(2/np.pi))
        if r < 1:
            r = 1
        if r > n:
            print('The failure probability is too low, the rank is set to the maximum.')
            r = n

        # Sketch initial random matrix
        np.random.seed(seed)
        Omega = np.random.randn(n, r)
        if do_ortho_sketch:
            Omega = la.orth(Omega)
        Omega.astype(A0.dtype)
        
        # Initial approximation for the range
        j = 1
        if do_ortho_sketch:
            # Omega = la.orth(np.column_stack([A0.U, Omega])) # Augmented basis
            problem.select_ode('K', mats_uv=(Omega,))
        else:
            Wh = la.solve(Omega.T.conj().dot(Omega), Omega.T.conj())
            problem.select_ode('B', mats_uv=(Omega, Wh))
        B0 = A0.dot(Omega, dense_output=True)
        B1 = solve_matrix_ivp(problem, t_subspan, B0, dense_output=True, **substep_kwargs)
        Q, R = la.qr(B1, mode='economic') 

        # Adaptive loop
        current_max = np.inf
        while current_max > tol:
            # Draw a new sketch matrix
            Omega = np.random.randn(n, r)
            if do_ortho_sketch:
                Omega = la.orth(Omega)
            Omega.astype(A0.dtype)

            # Compute the new approximation
            if do_ortho_sketch:
                problem.select_ode('K', mats_uv=(Omega,))
            else:
                Wh = la.solve(Omega.T.conj().dot(Omega), Omega.T.conj())
                problem.select_ode('B', mats_uv=(Omega, Wh))
            B0 = A0.dot(Omega, dense_output=True)
            B1 = solve_matrix_ivp(problem, t_subspan, B0, dense_output=True, **substep_kwargs)

            # Compute the error estimated on the new samples
            E = B1 - Q.dot(Q.T.conj().dot(B1))
            current_max = np.max(np.linalg.norm(E, axis=0))

            # If error is too large -> update the approximation
            if current_max > tol:
                j += 1
                Q, R = la.qr_insert(Q, R, E, -1, which='col') # Update QR decomposition is fast

        return Q


#%% The dynamical co-rangefinder method
def dynamical_corangefinder(matrix_ode: MatrixOde,
                            t_subspan: tuple, 
                            Y0: LowRankMatrix,
                            substep_kwargs: dict = {'solver': 'scipy'}, 
                            target_rank: int = 0,
                            oversampling_parameter: int = 10,
                            power_iterations: int = 1,
                            do_ortho_sketching: bool = True,
                            seed: int = 1234) -> Any:
        """
        The dynamical co-range finder method.

        Parameters
        ----------
        matrix_ode : MatrixOde
            The matrix ODE with selection of the fields: 'K' and 'L' for orthogonal sketching or 'B' and 'C' for non-orthogonal sketching.
        t_subspan : tuple
            The time span for the integration.
        Y0 : LowRankMatrix
            The initial value as a LowRankMatrix.
        substep_kwargs : dict, optional
            The solver for the substep, by default {'solver': 'scipy'}.
        target_rank : int, optional
            The target rank of the approximation, by default 0.
        oversampling_parameter : int, optional
            The oversampling parameter, by default 10.
        power_iterations : int, optional
            The number of power iterations, by default 1.
        do_ortho_sketching : bool, optional
            If the sketch matrix is orthogonal, by default True.
        seed : int, optional
            The seed for the random number generator, by default 1234.
            
        Returns
        -------
        Q: np.ndarray
            A matrix with orthogonal columns approximating the co-range of the ODE at the final time. Size m x (r+p).
        """

        # CHECK INPUTS
        assert len(t_subspan) == 2, 't_subspan must be a tuple of length 2.'
        assert isinstance(Y0, (np.ndarray, QuasiSVD)), 'initial_value must be an array or a QuasiSVD (or SVD).'

        # INITIALISATION
        A0 = Y0
        (m, n) = Y0.shape
        problem: MatrixOde = matrix_ode
        if target_rank == 0:
            warnings.warn('The target rank is not provided, so it is set to the rank of the initial value.')
            target_rank = Y0.rank
        
        # SKETCH RANDOM MATRIX
        np.random.seed(seed)
        Omega = np.random.randn(m, target_rank + oversampling_parameter)
        if do_ortho_sketching:
            Omega = la.orth(Omega)
            is_ortho = True
        else:
            is_ortho = False

        # C-STEP: FIND THE CO-RANGE OF THE ODE
        if is_ortho == True:
            # Omega = la.orth(np.column_stack([A0.V, Omega])) # Augmented basis
            problem.select_ode('L', mats_uv=(Omega,))
        else:
            Zh = la.solve(Omega.T.conj().dot(Omega), Omega.T.conj())
            problem.select_ode('C', mats_uv=(Omega, Zh))
        C0 = A0.T.conj().dot(Omega, dense_output=True)
        C1 = solve_matrix_ivp(problem, t_subspan, C0, dense_output=True, **substep_kwargs)
        Q, _, _ = la.qr(C1, mode='economic', pivoting=True) # QR decomposition

        # Power iterations
        for _ in range(power_iterations):
            # B-STEP
            B0 = A0.dot(Q, dense_output=True)
            problem.select_ode('K', mats_uv=(Q,))
            B1 = solve_matrix_ivp(problem, t_subspan, B0, dense_output=True, **substep_kwargs)
            Q, _, _ = la.qr(B1, mode='economic', pivoting=True)

            # C-STEP
            C0 = A0.T.conj().dot(Q, dense_output=True)
            problem.select_ode('L', mats_uv=(Q,))
            C1 = solve_matrix_ivp(problem, t_subspan, C0, dense_output=True, **substep_kwargs)
            Q, _, _ = la.qr(C1, mode='economic', pivoting=True)

        return Q

#%% The adaptive dynamical co-rangefinder method
def adaptive_dynamical_corangefinder(matrix_ode: MatrixOde,
                                     t_subspan: tuple, 
                                     Y0: LowRankMatrix,
                                     substep_kwargs: dict = {'solver': 'scipy'},
                                     tol: float = 1e-8,
                                     failure_prob: float = 1e-6,
                                     power_iterations: int = 1,
                                     do_ortho_sketching: bool = False,
                                     seed: int = 1234) -> Any:
        """
        The adaptive dynamical co-range finder method.

        The tolerance corresponds to the error made by the approximation of the co-range of the ODE at the final time : 
        ||A(h)^H - Q Q^H A(h)^H||_F <= tol
        The failure probability is the probability of not satisfying the tolerance.
        Low failure probability implies larger sampling method.
        The error is estimated with the samples used during the method.

        Reference: It is a new algorithm, not yet published.

        Parameters
        ----------
        matrix_ode : MatrixOde
            The matrix ODE with selection of the fields: 'K' and 'L' for orthogonal sketching or 'B' and 'C' for non-orthogonal sketching.
        t_subspan : tuple
            The time span for the integration.
        Y0 : LowRankMatrix
            The initial value as a LowRankMatrix of size m x n.
        substep_kwargs : dict, optional
            The solver for the substep, by default {'solver': 'scipy'}.
        tol : float, optional
            The tolerance for the adaptive method, by default 1e-8.
        failure_prob : float, optional
            The failure probability for the adaptive method, by default 1e-6.
        do_ortho_sketching : bool, optional
            If the sketch matrix is orthogonal, by default False.
            
        Returns
        -------
        Q: np.ndarray
            A matrix with orthogonal columns approximating the co-range of the ODE at the final time. Size m x (r+p).
        """

        # CHECK INPUTS
        assert len(t_subspan) == 2, 't_subspan must be a tuple of length 2.'
        assert isinstance(Y0, (np.ndarray, QuasiSVD)), 'initial_value must be an array or a QuasiSVD (or SVD).'

        # INITIALISATION
        A0 = Y0
        (m, n) = A0.shape
        problem: MatrixOde = matrix_ode
        q = power_iterations

        # Compute the sketch size according to the failure probability
        n = min(A0.shape)
        r = int(np.ceil(- np.log(failure_prob) / np.log(10)))
        tol = tol / (10 * np.sqrt(2/np.pi))
        if r < 1:
            r = 1
        if r > n:
            print('The failure probability is too low, the rank is set to the maximum.')
            r = n

        # Draw first r random vectors
        np.random.seed(seed)
        Omega = np.random.randn(n, r)
        if do_ortho_sketching:
            Omega = la.orth(Omega)
        Omega.astype(A0.dtype)
        
        # Initial approximation for the co-range
        if do_ortho_sketching:
            # Omega = la.orth(np.column_stack([A0.V, Omega])) # Augmented basis
            problem.select_ode('L', mats_uv=(Omega,))
        else:
            Zh = la.solve(Omega.T.conj().dot(Omega), Omega.T.conj())
            problem.select_ode('C', mats_uv=(Omega, Zh))
        C0 = A0.T.conj().dot(Omega, dense_output=True)
        C1 = solve_matrix_ivp(problem, t_subspan, C0, dense_output=True, **substep_kwargs)
        Q, R = la.qr(C1, mode='economic')

        # Adaptive method loop
        current_max = np.inf
        j = 1
        while current_max > tol:
            # Draw a new sketch matrix
            Omega = np.random.randn(n, r)
            if do_ortho_sketching:
                Omega = la.orth(Omega)
            Omega.astype(A0.dtype)

            # Compute the new approximation
            if do_ortho_sketching:
                problem.select_ode('L', mats_uv=(Omega,))
            else:
                Zh = la.solve(Omega.T.conj().dot(Omega), Omega.T.conj())
                problem.select_ode('C', mats_uv=(Omega, Zh))
            C0 = A0.T.conj().dot(Omega, dense_output=True)
            C1 = solve_matrix_ivp(problem, t_subspan, C0, dense_output=True, **substep_kwargs)


            # Compute the error estimated on the new samples
            E = C1 - Q.dot(Q.T.conj().dot(C1))
            current_max = np.max(np.linalg.norm(E, axis=0))

            # Augment the basis if the error is too large
            if current_max > tol:
                j += 1
                Q, R = la.qr_insert(Q, R, E, -1, which='col')

        return Q
