"""
Author: Benjamin Carrel, University of Geneva, 2022

This file contains the class MatrixOdeSolver, which is a general class for solving matrix ODEs
"""

#%% Imports
import numpy as np
from numpy import ndarray
from low_rank_toolbox import SVD, LowRankMatrix
from matrix_ode_toolbox import MatrixOde

Matrix = ndarray | LowRankMatrix


#%% Class MatrixOdeSolver
class MatrixOdeSolver:
    """
    Matrix ODE solver class.

    How to define a new solver:
    1. Define a new class that inherits from MatrixOdeSolver.
    2. Overload the __init__ method to add the necessary arguments.
    3. Overload the stepper method to define the solver.
    4. Overload the info property to return the solver information.

    See subclasses for examples.
    """

    #%% ATTRIBUTES
    name: str = "Matrix ODE solver"

    #%% Static methods
    def __init__(self, matrix_ode: MatrixOde, nb_substeps: int, **kwargs):
        """
        Initialize the matrix solver.

        Parameters
        ----------
        matrix_ode : MatrixOde
            The matrix ODE to solve
        nb_substeps : int
            The number of substeps to use in solve()
        """
        self.matrix_ode = matrix_ode
        self.nb_substeps = nb_substeps
        self.extra_data = kwargs


    @property
    def info(self) -> str:
        "Return the info string."
        info = f"{self.name} \n"
        info += f"-- {self.nb_substeps} substep(s)"
        return info

    def __repr__(self):
        return self.info

    def solve(self, t_span: tuple, X0: Matrix):
        """
        Solve the matrix ODE from t0 to t1, with initial value X0.
        Applies the stepper method nb_substeps times, and returns the final value.
        """
        # VARIABLES
        t0, t1 = t_span
        ts = np.linspace(t0, t1, self.nb_substeps + 1, endpoint=True)
        X = X0
        # LOOP
        for i in np.arange(self.nb_substeps):
            X = self.stepper(ts[i:i+2], X)
        return X

    def solve_best_rank(
        self,
        t_span: tuple,
        X0: Matrix,
        rank: int,
        rtol: float = None,
        dense_output: bool = False,
    ) -> Matrix:
        """
        Solve the problem truncated to the given rank.

        Parameters
        ----------
        rank : int
            The rank of the low rank approximation. Truncate the solution to this rank. By default, no truncation.
        rtol : float, optional
            The relative tolerance for the truncation. By default None.
        dense_output : bool, optional
            Whether to return a dense matrix, by default False

        Returns
        -------
        Matrix
            The solution
        """
        # Check the rank of the initial value and truncate if necessary
        if isinstance(X0, LowRankMatrix):
            if X0.rank < rank:
                print(
                    f"Warning: the initial value has rank {X0.rank}, which is smaller than the desired rank {rank}."
                )
            if X0.rank > rank:
                X0 = SVD.from_low_rank(X0).truncate(rank, rtol=rtol)
        else:
            X0 = SVD.truncated_svd(X0, rank, rtol=rtol)
        # Solve
        X1 = self.solve(t_span, X0, **self.extra_args)
        # Truncate the solution if necessary
        if isinstance(X1, LowRankMatrix):
            if X1.rank > rank:
                X1 = SVD.truncated_svd(X1, rank, rtol=rtol)
        else:
            X1 = SVD.truncated_svd(X1, rank, rtol=rtol)
        if dense_output:
            X1 = X1.todense()
        return X1

    #%% Methods to be overloaded
    def stepper(self, t_span: tuple, X0: Matrix) -> Matrix:
        """
        Compute the next step of the solution.
        Overload this method to define a new solver.

        Parameters
        ----------
        t_span : tuple
            The time interval (t0, t1)
        X0 : Matrix
            The initial value at t0
        extra_args : dict
            Extra arguments

        Returns
        -------
        Matrix
            The solution at t1
        """
        raise NotImplementedError(
            "Stepper is not implemented for the generic solver. Stepper must be overloaded."
        )


# class MatrixOdeSolution:
#     """
#     Store the solution of a matrix IVP.

#     Additionaly, this class can perform some basic operations on the solution.
#     """

#     # %% BASIC FUNCTIONS
#     def __init__(self,
#                  problem: GeneralIVP,
#                  ts: list,
#                  Xs: list,
#                  time_of_computation: list):
#         # STORE DATA
#         self.problem = problem
#         self.ts = ts
#         self.Xs = Xs
#         self.time_of_computation = time_of_computation

#     def __repr__(self):
#         return (
#             f"IVP: {self.problem_name} of shape {self.shape}\n"
#             f"Number of time steps: {self.nb_t_steps} \n"
#             f"Stepsize: h={self.stepsize} \n"
#             f"Total time of computation: {round(np.sum(self.time_of_computation), 2)} seconds"
#         )

#     # %% PROPERTIES
#     @property
#     def shape(self) -> tuple:
#         return self.Xs[0].shape

#     @property
#     def nb_t_steps(self) -> int:
#         return len(self.ts)

#     @property
#     def stepsize(self) -> float:
#         return self.ts[1] - self.ts[0]

#     @property
#     def h(self) -> float:
#         return self.stepsize

#     def copy(self):
#         return MatrixOdeSolution(self.problem, self.ts, self.Xs, self.time_of_computation)

#     def convert_to_dense(self):
#         old_sol = self.Xs
#         for k, Y in enumerate(old_sol):
#             if isinstance(Y, LowRankMatrix):
#                 self.Xs[k] = Y.todense()

#     def convert_to_SVD(self):
#         old_sol = self.Xs
#         for k, Y in enumerate(old_sol):
#             if not isinstance(Y, SVD):
#                 self.Xs[k] = SVD.truncated_svd(Y)

#     def plot(self,
#              time_index: str,
#              title: str = None,
#              do_save: bool = False,
#              filename: str = None):
#         "Plot the solution corresponding to the time index"
#         # VARIABLES
#         time = self.ts[time_index]
#         solution = self.Xs[time_index]
#         if isinstance(title, NoneType):
#             title = f'{self.problem_name} - Solution at time t={round(self.ts[time_index], 2)}'
#         return self.problem.imshow(solution, title, do_save, filename)

#     def plot_singular_values(self,
#                              time_index: str,
#                              title: str = None,
#                              do_save: bool = False,
#                              filename: str = None):
#         "Plot the singular values of the solution corresponding to the time index"
#         # VARIABLES
#         solution = self.Xs[time_index]
#         if isinstance(solution, SVD):
#             sing_vals = solution.sing_vals
#         else:
#             sing_vals = la.svd(solution, compute_uv=False)
#         index = np.arange(1, len(sing_vals) + 1)
#         if isinstance(title, NoneType):
#             title = f'{self.problem_name} - Singular values at time t={round(self.ts[time_index], 2)}'

#         # PLOT
#         fig = plt.figure(clear=True)
#         plt.semilogy(index, sing_vals, 'o')
#         plt.semilogy(index, sing_vals[0] * np.finfo(float).eps * np.ones(len(sing_vals)), 'k--')
#         plt.title(title, fontsize=16)
#         plt.xlim((index[0], index[-1]))
#         plt.xlabel('index', fontsize=16)
#         plt.ylim((1e-18, 1e2))
#         plt.ylabel('value', fontsize=16)
#         plt.tight_layout()
#         plt.grid()
#         plt.show()
#         if do_save:
#             fig.savefig(filename)
#         return fig

#     def plot_solution(self,
#                       time_index: str,
#                       title: str = None,
#                       do_save: bool = False,
#                       filename: str = None):
#         "Plot the solution with the representation of the problem at the given time index"
#         # VARIABLES
#         solution = self.Xs[time_index]
#         if isinstance(title, NoneType):
#             title = f'{self.problem_name} - Solution at time t={round(self.ts[time_index], 2)}'

#         # PLOT
#         fig = self.problem.imshow(solution, title, do_save, filename)
#         return fig


#     def animation_singular_values(self,
#                                 title: str = None,
#                                 do_save: bool = False) -> animation.FuncAnimation:
#         "Return an animation of the singular values"
#         return plotting.singular_values(self.ts, self.Xs, title, do_save)

#     def animation_2D(self,
#                      title: str = None,
#                      do_save: bool = False):
#         "Return an animation in 2D of the solution"
#         return plotting.animation_2D(self.ts, self.Xs, title, do_save)

#     def animation_sing_vals_and_sol(self,
#                                     title: str = None,
#                                     do_save: bool = False):
#         "Return an animation of the singular values together with the problem's representation"


#     # %% ERRORS
#     def compute_errors(self, other) -> ndarray:
#         "Compute error at each time step with an other solution."
#         # SOME VERIFICATIONS
#         checkers = (self.shape == other.shape,
#                     self.nb_t_steps == other.nb_t_steps)
#         if not all(checkers):
#             raise ValueError('The two solutions do not corresponds')
#         # COMPUTE ERRORS ITERATIVELY
#         errors = np.zeros(self.nb_t_steps)
#         for k in np.arange(self.nb_t_steps):
#             if isinstance(other.Ys[k], LowRankMatrix):
#                 diff = other.Ys[k] - self.Xs[k]
#             else:
#                 diff = self.Xs[k] - other.Ys[k]
#             if isinstance(diff, LowRankMatrix):
#                 errors[k] = diff.norm()
#             else:
#                 errors[k] = la.norm(diff)
#         return errors

#     def compute_relative_errors(self, ref_sol) -> ndarray:
#         "Compute relative error at each time step with an other solution."
#         # SOME VERIFICATIONS
#         checkers = (self.shape == ref_sol.shape,
#                     self.nb_t_steps == ref_sol.nb_t_steps)
#         if not all(checkers):
#             raise ValueError('The two solutions do not corresponds')
#         # COMPUTE ERRORS ITERATIVELY
#         errors = np.zeros(self.nb_t_steps)
#         for k in np.arange(self.nb_t_steps):
#             if isinstance(ref_sol.Ys[k], LowRankMatrix):
#                 normYs = ref_sol.Ys[k].norm()
#                 diff = ref_sol.Ys[k] - self.Xs[k]
#             else:
#                 normYs = la.norm(ref_sol.Ys[k])
#                 diff = self.Xs[k] - ref_sol.Ys[k]
#             if isinstance(diff, LowRankMatrix):
#                 errors[k] = diff.norm() / normYs
#             else:
#                 errors[k] = la.norm(diff) / normYs
#         return errors


# %% SINGULAR VALUE
