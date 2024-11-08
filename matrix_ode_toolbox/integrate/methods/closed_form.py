"""
Author: Benjamin Carrel, University of Geneva, 2022

Closed form solver. This is a solver for matrix ODEs with a closed form solution.
"""

#%% Imports
import numpy as np
import scipy.linalg as la
from matrix_ode_toolbox.integrate import MatrixOdeSolver
from low_rank_toolbox import LowRankMatrix
from numpy import ndarray
from scipy.sparse import spmatrix
import scipy.sparse.linalg as spsla
from krylov_toolbox import solve_sylvester, solve_lyapunov

Matrix = ndarray | LowRankMatrix | spmatrix

#%% Class
class ClosedFormSolver(MatrixOdeSolver):
    """
    Class to solve a matrix ODE using the closed form solution.

    How to implement a new closed form solution:
    1. Define a new method in this class, which takes t_span, X0, and extra arguments.
    2. Add the method to the stepper method with an elif statement.
    Well done! You can now use your new closed form solution.
    """

    #%% ATTRIBUTES
    name: str = 'Closed form'
    supported_odes: list[str] = ['Sylvester', 'Lyapunov', 'Cookie']

    def __init__(self, matrix_ode, nb_substeps:  int = 1, **closed_form_args):
        super().__init__(matrix_ode, nb_substeps)
        # Specific closed form arguments
        if matrix_ode.name == 'Sylvester' or matrix_ode.name == 'Lyapunov':
            if 'traceA' not in closed_form_args:
                closed_form_args['traceA'] = matrix_ode.A.trace()
            if 'traceB' not in closed_form_args:
                closed_form_args['traceB'] = matrix_ode.B.trace()
        # Store the closed form arguments
        self.closed_form_args = closed_form_args

        # Preprocessing
        if matrix_ode.name == 'Cookie':
            C = matrix_ode.C1r
            if isinstance(C, spmatrix): # sanity check
                C = C.todense()
            self.L, self.Q = la.eigh(C, eigvals_only=False)

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Closed form solver \n'
        info += f'-- {self.nb_substeps} substep(s)'
        return info

    def stepper(self, t_span: tuple, X0: Matrix) -> Matrix:
        """
        Automatically choose the closed form solution based on the current ode.
        """
        if self.matrix_ode.name == 'Sylvester':
            Ar, Br, Cr = self.matrix_ode.Ar, self.matrix_ode.Br, self.matrix_ode.Cr
            return self.closed_form_invertible_diff_sylvester(t_span, X0, Ar, Br, Cr, **self.closed_form_args)
        elif self.matrix_ode.name == 'Lyapunov':
            Ar, Cr = self.matrix_ode.Ar, self.matrix_ode.Cr
            return self.closed_form_invertible_diff_lyapunov(t_span, X0, Ar, Cr, **self.closed_form_args)
        elif self.matrix_ode.name == 'Cookie':
            A, B, D = - self.matrix_ode.A0r, - self.matrix_ode.A1r, self.matrix_ode.Br
            L, Q = self.L, self.Q
            if self.matrix_ode.ode_type == 'L':
                return self.closed_form_cookie(t_span, X0.T, A, B, L, Q, D.T).T
            else:
                return self.closed_form_cookie(t_span, X0, A, B, L, Q, D)

        else:
            raise NotImplementedError(f"The closed form solution for {self.matrix_ode.name} is not implemented yet.")

    @classmethod
    def closed_form_invertible_diff_sylvester(cls, t_span: tuple, X0: Matrix, A: Matrix, B: Matrix, C: Matrix, **kwargs) -> Matrix:
        """Closed form solution of the sylvester differential equation.
        X' = AX + XB + C
        X(t0) = X0
        
        The solution is:
        X(t) = exp((t-t0)A) (X0 + Z) exp((t-t0)B) - Z
        where AZ + ZB = C

        Parameters
        ----------
        t_span : tuple
            Time interval
        X0 : Matrix
            Initial value, shape (m,n)
        A : Matrix
            Matrix A of the sylvester equation (m, m)
        B : Matrix
            Matrix B of the sylvester equation (n, n)
        C : Matrix
            Matrix C of the sylvester equation (m, n)

        Returns
        -------
        Matrix
            Solution of the sylvester differential equation at final time
        """
        # INITIALIZATION
        (t0, t1) = t_span
        h = t1 - t0

        # EXTRACT ARGUMENTS FROM kwargs OR COMPUTE THEM
        # Trace of A and B
        if 'traceA' in kwargs:
            traceA = kwargs['traceA']
        else:
            traceA = A.trace()
        if 'traceB' in kwargs:
            traceB = kwargs['traceB']
        else:
            traceB = B.trace()
        
        # SOLVE SYLVESTER SYSTEM
        Z = solve_sylvester(A, B, C, **kwargs)
        if isinstance(Z, LowRankMatrix):
            M = Z + X0
        else:
            M = X0 + Z
        
        # COMPUTE MATRIX EXPONENTIAL
        # Transform A and B to sparse matrices if they are not already
        if not isinstance(A, spmatrix):
            A = spsla.aslinearoperator(A)
        if not isinstance(B, spmatrix):
            B = spsla.aslinearoperator(B)
        if isinstance(M, LowRankMatrix):
            N = M.expm_multiply(A, h, side='left', traceA=traceA)
            Y = N.expm_multiply(B, h, side='right', traceA=traceB)
        else:
            N = spsla.expm_multiply(A, M, start=0, stop=h, endpoint=True, num=2, traceA=traceA)[-1]
            Y = spsla.expm_multiply(B.T, N.T, start=0, stop=h, endpoint=True, num=2, traceA=traceB)[-1].T
        
        # ASSEMBLE SOLUTION
        if isinstance(Z, LowRankMatrix):
            X1 = Z - Y
        else:
            X1 =  Y - Z
        return X1

    @classmethod
    def closed_form_invertible_diff_lyapunov(cls, t_span: tuple, X0: Matrix, A: Matrix, C: Matrix, **kwargs) -> Matrix:
        """Closed form of the differential Lyapunov equation.
        X' = AX + XA + C
        X(t0) = X0
        
        The solution is:
        X(t) = exp(A(t-t0)) (X0 + Z) exp(A(t-t0)) - Z
        where AZ + ZA = C

        Parameters
        ----------
        t_span : tuple
            The time interval
        X0 : Matrix
            The initial value
        A : Matrix
            The matrix A
        C : Matrix
            The matrix C
        invA : object, optional
            The inverse of A, by default None
        """
        # INITIALIZATION
        t0, t1 = t_span
        h = t1 - t0

        # EXTRACT ARGUMENTS FROM kwargs OR COMPUTE THEM
        if 'traceA' in kwargs:
            traceA = kwargs['traceA']
        else:
            traceA = A.trace()

        # SOLVE LYAPUNOV SYSTEM
        Z = solve_lyapunov(A, C, **kwargs)
        M = Z + X0 

        # COMPUTE MATRIX EXPONENTIAL
        if isinstance(M, LowRankMatrix):
            N = M.expm_multiply(A, h, side='left')
            Y = N.expm_multiply(A, h, side='right')
        else:
            N = spsla.expm_multiply(A, M, start=0, stop=h, endpoint=True, num=2, traceA=traceA)[-1]
            Y = spsla.expm_multiply(A.T, N.T, start=0, stop=h, endpoint=True, num=2, traceA=traceA)[-1].T
        
        # ASSEMBLE SOLUTION
        # try:
        #     X1 = Y - Z
        # except:
        #     X1 = Z - Y
        # print(Y)
        # print(Z)
        X1 = Y - Z
        return X1

    @classmethod
    def closed_form_cookie(cls, 
                        t_span: tuple,
                        X0: Matrix,
                        A: Matrix,
                        B: Matrix,
                        L: Matrix,
                        Q: Matrix,
                        D: Matrix) -> Matrix:
        """
        Return the solution of the problem:
        X' = AX + BXC + D
        where C = Q * L * Q^T (C is supposed to be symmetric)
        X(t0) = X0
        See my master thesis for details.

        Parameters
        ----------
        t_span: tuple
            initial and final time (t0,t1)
        X0: ndarray
            initial value at ts t0
        A, B, Q, L, D: ndarray
            matrices of the problem as described above

        Returns
        -------
        X1: ndarray
            solution at time t1
        """

        # Creation of variables
        h = t_span[1] - t_span[0]
        n, p = D.shape
        m1 = np.zeros(n * p)
        m2 = np.zeros(n * p)
        m3 = np.zeros(n * p)

        if isinstance(X0, LowRankMatrix):
            X0 = X0.todense()

        # Efficient computation of the solution
        D_tilde = D.dot(Q)
        X0_tilde = X0.dot(Q)
        for i in range(p):
            Ai = A + L[i] * B
            if isinstance(Ai, spmatrix):
                m1[i * n: (i + 1) * n] = spsla.spsolve(Ai, D_tilde[:, i])
            else:
                m1[i * n: (i + 1) * n] = la.solve(Ai, D_tilde[:, i]).reshape(-1)
            m2[i * n: (i + 1) * n] = spsla.expm_multiply(
                Ai, m1[i * n: (i + 1) * n], start=0, stop=h, num=2
            )[-1]
            m3[i * n: (i + 1) * n] = spsla.expm_multiply(
                Ai, X0_tilde[:, i], start=0, stop=h, num=2
            )[-1]

        M1 = m1.reshape(D.shape, order="F")
        M2 = m2.reshape(D.shape, order="F")
        M3 = m3.reshape(D.shape, order="F")
        X1 = (M3 + M2 - M1).dot(Q.T)
        return X1

