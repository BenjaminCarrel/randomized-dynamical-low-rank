"""
Author: Benjamin Carrel, University of Geneva, 2022

Scipy wrapper to solve the DLRA. 
This method is inefficient but can be used as a reference solution of the DLRA.
"""

#%% Imports
from numpy import ndarray
from matrix_ode_toolbox.dlra import DlraSolver
from scipy.integrate import solve_ivp
from matrix_ode_toolbox import MatrixOde
from low_rank_toolbox import LowRankMatrix, SVD
from numpy import ndarray

Matrix = LowRankMatrix | ndarray

#%% Class ScipyDlra
class ScipyDlra(DlraSolver):
    """
    Class ScipyDlra.
    The method is a wrapper around scipy.integrate.solve_ivp.
    It solves the problem orthogonally projected onto the matrix manifold, which is the DLRA.
    The method is inefficient and might be unstable, but can be used as a reference solution of the DLRA for small problems.
    NOTE: Apply this method only on small problems, for testing purposes.
    """

    def __init__(self, 
                 matrix_ode: MatrixOde, 
                 nb_substeps: int = 1, 
                 scipy_kwargs: dict = {'solver': 'RK45', 'rtol': 1e-12, 'atol': 1e-12},
                 **extra_args) -> None:
        super().__init__(matrix_ode, nb_substeps)
        # Save the kwargs
        self.scipy_kwargs = scipy_kwargs
        self.extra_data['sols'] = []

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'DLRA solved by scipy (for testing purposes only on small problems) \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f'-- {self.scipy_kwargs["solver"]} solver'
        return info

    def stepper(self, t_subspan: tuple, Y0: Matrix) -> SVD:
        "Solves Y' = P(Y)[F(Y)] using scipy.integrate.solve_ivp."
        # Check inputs
        assert len(t_subspan) == 2, "t_subspan must be a tuple of length 2."
        assert isinstance(Y0, Matrix), "Y0 must be a LowRankMatrix (or np.ndarray)."

        # Initialisation
        rank = Y0.rank
        shape = Y0.shape
        y0 = Y0.todense().flatten()

        # Vectorized function
        def vec_f(t: float, y: ndarray) -> ndarray:
            Y = SVD.truncated_svd(y.reshape(shape), rank)
            dY = self.matrix_ode.tangent_space_ode_F(t, Y)
            return dY.todense().flatten()

        # Solve the ODE
        sol = solve_ivp(vec_f, t_subspan, y0, dense_output=True, **self.scipy_kwargs)
        y1 = sol.y[:, -1]
        self.extra_data['sols'] += [sol]
        Y1 = SVD.truncated_svd(y1.reshape(shape), rank)
        return Y1
