"""
Author: Benjamin Carrel, University of Geneva, 2022

Projected Runge-Kutta methods for the DLRA.
See Kieri & Vandereycken 2019.
"""

# %% Imports
from low_rank_toolbox import QuasiSVD, SVD
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




#%% Class Projected Runge Kutta
class ProjectedRungeKutta(DlraSolver):
    """
    Class for the projected Runge-Kutta DLRA methods.
    See Kieri and Vandereycken, 2019.

    """

    name = 'Projected Runge-Kutta (PRK)'
    _allow_automatic_truncation = False

    def __init__(self,
                matrix_ode: MatrixOde,
                nb_substeps: int = 1,
                order: int = 2,
                **extra_kwargs) -> None:
        super().__init__(matrix_ode, nb_substeps)
        self.order = order

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Projected Runge-Kutta (PRK) (Kieri & Vandereycken 2019) \n'
        info += f'-- {self.order} stage(s) \n'
        info += f'-- {self.nb_substeps} substep(s)'
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

    def stepper(self, t_subspan: tuple, Y0: QuasiSVD) -> SVD:
        """
        One-step method of projected Runge-Kutta of the given order.

        Parameters
        ----------
        t_subspan : tuple
            The time interval (t0, tf).
        Y0 : QuasiSVD
            The initial value. 
        """
        # Check inputs
        assert len(t_subspan) == 2, "t_subspan must be a tuple of length 2."
        assert isinstance(Y0, QuasiSVD), "Y0 must be a QuasiSVD (or SVD)."
        
        # Variable
        rank = Y0.rank
        a, b, c = self.RK_rule
        s = self.order
        h = t_subspan[1] - t_subspan[0]
        eta = np.empty(s, dtype=SVD)
        kappa = np.empty(s, dtype=SVD)

        # PRK METHOD
        eta[0] = Y0
        kappa[0] = self.matrix_ode.tangent_space_ode_F(t_subspan[0], eta[0], truncate=self._allow_automatic_truncation)

        # PRK LOOP
        for j in np.arange(1, s):
            if self._allow_automatic_truncation:
                big_eta = Y0 + h * np.sum([a[j, i] * kappa[i] for i in np.arange(0, j)])
            else:
                big_eta = SVD.multi_add([Y0] + [h * a[j, i] * kappa[i] for i in np.arange(0, j)], truncate=self._allow_automatic_truncation)
            eta[j] = big_eta.truncate(rank)
            tj = t_subspan[0] + c[j] * h
            kappa[j] = self.matrix_ode.tangent_space_ode_F(tj, eta[j], truncate=self._allow_automatic_truncation)

        # PRK OUTPUT
        if self._allow_automatic_truncation:
            Y1 = Y0 + h * np.sum([b[i] * kappa[i] for i in np.arange(0, s)])
        else:
            Y1 = SVD.multi_add([Y0] + [h * b[i] * kappa[i] for i in np.arange(0, s)], truncate=self._allow_automatic_truncation)
        return Y1.truncate(rank)

