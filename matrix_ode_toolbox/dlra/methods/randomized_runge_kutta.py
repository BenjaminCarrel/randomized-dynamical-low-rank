"""
Author: Benjamin Carrel, University of Geneva, 2024

Randomized low-rank Runge-Kutta methods for solving matrix ODEs.
See Hysam Lam & Daniel Kressner 2024
CONFIDENTIAL - NOT PUBLISHED YET
"""

# %% Imports
from low_rank_toolbox import QuasiSVD, SVD, LowRankMatrix
import numpy as np
from matrix_ode_toolbox.dlra import DlraSolver
from matrix_ode_toolbox import MatrixOde


#%% Runge-Kutta tables
# ORDER 1
a1 = float(0)
b1 = np.ones(1)
# ORDER 2
a2 = np.zeros((2, 2))
a2[1, 0] = 1
b2 = np.zeros(2)
b2[0] = 1/2
b2[1] = 1/2
# ORDER 3
a3 = np.zeros((3, 3))
a3[1, 0] = 1/3
a3[2, 0] = 0
a3[2, 1] = 2/3
b3 = np.zeros(3)
b3[0] = 1/4
b3[2] = 3/4
# ORDER 4
a4 = np.zeros((4, 4))
a4[1, 0] = 1/2
a4[2, 1] = 1/2
a4[3, 2] = 1
b4 = np.zeros(4)
b4[0] = 1/6
b4[1] = 1/3
b4[2] = 1/3
b4[3] = 1/6
# Rule 6(5)9b
a8 = np.zeros((8, 8))
a8[1, 0] = 1/8
a8[2, 0] = 1/18
a8[3, 0] = 1/16
a8[4, 0] = 1/4
a8[5, 0] = 134/625
a8[6, 0] = -98/1875
a8[7, 0] = 9/50
a8[2, 1] = 1/9
a8[3, 2] = 3/16
a8[4, 2] = -3/4
a8[5, 2] = -333/625
a8[6, 2] = 12/625
a8[7, 2] = 21/25
a8[4, 3] = 1
a8[5, 3] = 476/625
a8[6, 3] = 10736/13125
a8[7, 3] = -2924/1925
a8[5, 4] = 98/625
a8[6, 4] = -1936/1875
a8[7, 4] = 74/25
a8[6, 5] = 22/21
a8[7, 5] = -15/7
a8[7, 6] = 15/22
b8 = np.zeros(8)
b8[0] = 11/144
b8[3] = 256/693
b8[5] = 125/504
b8[6] = 125/528
b8[7] = 5/72


# %% Randomized Runge-Kutta methods
class RandomizedRungeKutta(DlraSolver):
    """
    Class for the randomized Runge-Kutta methods.
    See Hysam Lam & Daniel Kressner, 2024. CONFIDENTIAL - NOT PUBLISHED YET
    """

    name = "Randomized Runge-Kutta"
    

    def __init__(self,
                matrix_ode: MatrixOde,
                nb_substeps: int = 1,
                order: int = 2,
                rank = 5,
                oversampling_p: int = 5,
                oversampling_l: int = 5,
                **extra_kwargs) -> None:
        super().__init__(matrix_ode, nb_substeps)
        self.order = order
        self.r = rank
        self.p = oversampling_p
        self.l = oversampling_l
        self.shape = matrix_ode.shape
        self.Omegas, self.Psis = self.draw_random_matrices(self.order + 1, self.shape)

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Randomized Runge-Kutta (Lam & Kressner 2024) \n'
        info += f'-- {self.order} stage(s) \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f'-- Rank of the approximation: r={self.r} \n'
        info += f'-- Oversampling parameters: p={self.p}, l={self.l} \n'
        info += f'-- Disclaimer: the implementation is not optimized - do not perform timing tests with this version.'
        return info
    
    @property
    def RK_rule(self) -> tuple:
        """Shortcut for calling the table"""
        s = self.order
        if s == 1:
            a = a1
            b = b1
        elif s == 2:
            a = a2
            b = b2
        if s == 3:
            a = a3
            b = b3
        if s == 4:
            a = a4
            b = b4
        if s == 8:
            a = a8
            b = b8
        c = np.zeros(s)
        for i in np.arange(1, s):
            c[i] = sum(a[i, j] for j in np.arange(1, i))
        return a, b, c
    
    def draw_random_matrices(self, nb_matrices: int, shape:tuple):
        """
        Draw once the random matrices to be used in the schem.
        """
        np.random.seed(2222)
        (m,n) = shape
        Omegas = []
        for i in range(nb_matrices):
            Omega_i = np.random.randn(n, self.r + self.p)
            Omegas.append(Omega_i)
        Psis = []
        for i in range(nb_matrices):
            Psi_i = np.random.randn(m, self.r + self.p + self.l)
            Psis.append(Psi_i)
        return Omegas, Psis
    
    def nystrom_process(self, K_hat: np.ndarray, K_tilde: np.ndarray, Omega: np.ndarray) -> QuasiSVD:
        """
        Post-processing of the Nystroem method.
        """
        # Stable version of the Nystroem post-processing
        C = SVD.truncated_svd(K_tilde.dot(Omega), r=self.r, rtol=None, atol=None)
        U = K_hat.dot(C.V)
        S = np.linalg.inv(C.S)
        V = K_tilde.T.dot(C.U)
        return QuasiSVD(U, S, V)

    # This is working
    def stepper_full(self, t_subspan:tuple, Y0: QuasiSVD) -> SVD:
        "Naive non-optimized version of the stepper - for testing purposes"
        # Check inputs
        assert len(t_subspan) == 2, "t_subspan must be a tuple of length 2."
        assert isinstance(Y0, QuasiSVD), "Y0 must be a QuasiSVD (or SVD)."
        assert Y0.rank == self.r, "The rank of Y0 must be equal to the rank of the method."

        # Variables
        rank = Y0.rank
        a, b, c = self.RK_rule
        s = self.order
        h = t_subspan[1] - t_subspan[0]
        eta = np.empty(s, dtype=QuasiSVD)
        kappa = np.empty(s, dtype=QuasiSVD)
        Neta = np.empty(s, dtype=QuasiSVD)

        # Fix the seed for the Gaussian matrices
        np.random.seed(2222)
        seeds = np.random.randint(0, 10000, size=s+1)

        # Initialization
        eta[0] = Y0
        Neta[0] = SVD.generalized_nystroem(Y0, rank, (self.p, self.l), seed=seeds[0]).truncate(rank)
        kappa[0] = self.matrix_ode.ode_F(t_subspan[0], Neta[0])

        # Loop
        for j in np.arange(1, s):
            eta[j] = Y0 + h * np.sum([a[j, i] * kappa[i] for i in np.arange(0, j)], axis=0)
            Neta[j] = SVD.generalized_nystroem(eta[j], rank, (self.p, self.l), seed=seeds[j]).truncate(rank)
            tj = t_subspan[0] + c[j] * h
            kappa[j] = self.matrix_ode.ode_F(tj, Neta[j])
        
        # Output
        Z1 = Y0 + h * np.sum([b[i] * kappa[i] for i in np.arange(0, s)], axis=0)
        Y1 = SVD.generalized_nystroem(Z1, rank, (self.p, self.l), seed=seeds[s]).truncate(rank)
        return Y1
    
    # NOT WORKING YET
    def stepper(self, t_subspan:tuple, Y0: QuasiSVD) -> SVD:
        """
        Perform one step of the randomized Runge-Kutta method.

        Parameters
        ----------
        t_subspan: tuple
            The time interval for the step
        Y0: QuasiSVD
            The initial value for the step
        
        Returns
        -------
        Y1: SVD
            The result of the step
        """
        
       # Check inputs
        assert len(t_subspan) == 2, "t_subspan must be a tuple of length 2."
        assert isinstance(Y0, QuasiSVD), "Y0 must be a QuasiSVD (or SVD)."
        assert Y0.rank == self.r, "The rank of Y0 must be equal to the rank of the method."

        # Variables
        s = self.order
        h = t_subspan[1] - t_subspan[0]
        a, b, c = self.RK_rule

        Omegas = self.Omegas
        Psis = self.Psis
        Z_hat = np.zeros((self.order, self.shape[0], self.r + self.p)) # Will contain the Z_j Omega_j
        K_hat = np.empty((s, s+1), dtype=object) # Will contain the F_j Omega_j
        Z_tilde = np.zeros((self.order, self.r + self.p + self.l, self.shape[1])) # Will contain the Psi_j^T Z_j
        K_tilde = np.empty((s, s+1), dtype=object) # Will contain the Psi_j^T F_j
        NZ = np.empty(s, dtype=QuasiSVD)

        # Initialization
        Z_hat[0] = Y0.dot(Omegas[0], dense_output=True)
        Z_tilde[0] = Y0.dot(Psis[0].T, side='left', dense_output=True)
        NZ[0] = self.nystrom_process(Z_hat[0], Z_tilde[0], Omegas[0])
        for q in np.arange(0, s+1):
            K_hat[0, q] = self.matrix_ode.ode_F(t_subspan[0], NZ[0]).dot(Omegas[q])
            if isinstance(K_hat[0, q], LowRankMatrix):
                K_hat[0, q] = K_hat[0, q].todense()
            K_tilde[0, q] = self.matrix_ode.ode_F(t_subspan[0], NZ[0]).T.dot(Psis[q]).T
            if isinstance(K_tilde[0, q], LowRankMatrix):
                K_tilde[0, q] = K_tilde[0, q].todense()

        for j in np.arange(1, s):
            # Compute the K_j
            Z_hat[j] = Y0.dot(Omegas[j]) + h * np.sum([a[j, i] * K_hat[i, j] for i in np.arange(0, j)], axis=0)
            if isinstance(Z_hat[j], LowRankMatrix):
                Z_hat[j] = Z_hat[j].todense()
            # Compute the W_j
            Z_tilde[j] = Y0.dot(Psis[j].T, side='left') + h * np.sum([a[j, i] * K_tilde[i, j] for i in np.arange(0, j)], axis=0)
            if isinstance(Z_tilde[j], LowRankMatrix):
                Z_tilde[j] = Z_tilde[j].todense()
            # Compute the Nystrom approximation
            NZ[j] = self.nystrom_process(Z_hat[j], Z_tilde[j], Omegas[j])
            # Compute the two F_j
            Fj = self.matrix_ode.ode_F(t_subspan[0] + c[j] * h, NZ[j])
            for q in np.arange(j, s+1):
                K_hat[j, q] = Fj.dot(Omegas[q])
                if isinstance(K_hat[j, q], LowRankMatrix):
                    K_hat[j, q] = K_hat[j, q].todense()
                K_tilde[j, q] = Fj.T.dot(Psis[q]).T
                if isinstance(K_tilde[j, q], LowRankMatrix):
                    K_tilde[j, q] = K_tilde[j, q].todense()

        # Compute the output
        Y1_hat = Y0.dot(Omegas[s], dense_output=True) + h * np.sum([b[i] * K_hat[i, s] for i in np.arange(0, s)], axis=0)
        Y1_tilde = Y0.dot(Psis[s].T, side='left', dense_output=True) + h * np.sum([b[i] * K_tilde[i, s] for i in np.arange(0, s)], axis=0)
        Y1 = self.nystrom_process(Y1_hat, Y1_tilde, Omegas[s])
        return Y1
        