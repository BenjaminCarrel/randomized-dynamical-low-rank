"""
Author: Benjamin Carrel, University of Geneva, 2024

Vlasov-Poisson problems.
Reference: Lukas Einkemmer and Christian Lubich, see https://arxiv.org/pdf/1801.01103.
"""

#%% IMPORTATIONS
from __future__ import annotations
import numpy as np
from matrix_ode_toolbox.structures import VlasovPoissonOde
from matrix_ode_toolbox.utils import spacetime


#%% Electric field
def electric_field(rho, dx):
    dxE = np.ones_like(rho) - rho
    # Integration
    fftdxE= np.fft.fft(dxE)
    n =fftdxE.size
    freq = np.fft.fftfreq(n, d=dx)
    Einter = np.zeros_like(fftdxE)
    Einter[1:] = fftdxE[1:]/(1j*freq[1:]*2*np.pi)
    E=np.fft.ifft(Einter)
    return E

#%% Landau damping
def make_landau_damping(nx: int = 64, nv: int = 256, dx_order: int = 2, dv_order: int = 2):
    """
    Landau damping problem. Theoretical decay rate of the electric field: -0.153.

    Time interval: (0, 50).

    Parameters
    ----------
    nx : int, optional
        Number of spatial points, by default 64.
    nv : int, optional
        Number of velocity points, by default 256.
    dx_order : int, optional
        Order of the spatial derivative (2 or 4), by default 2.
    dv_order : int, optional
        Order of the velocity derivative (2 or 4), by default 2.

    Returns
    -------
    ode : VlasovPoissonOde
        Vlasov-Poisson ODE.
    A0 : ndarray
        Initial distribution function.
    """
    # Constant variables
    alpha = 1e-2
    k = 0.5

    # Space field
    xmin = 0.0
    xmax = 4 * np.pi
    dx = (xmax - xmin) / nx
    xs = np.linspace(xmin, xmax, nx)
    if dx_order == 2:
        Dx = spacetime.centered_1d_dx2(nx, dx, periodic=True).todense()
    elif dx_order == 4:
        Dx = spacetime.centered_1d_dx4(nx, dx, periodic=True).todense()

    # Velocity field
    vmin = -6
    vmax = 6
    dv = (vmax - vmin) / nv
    vs = np.linspace(vmin, vmax, nv)
    if dv_order == 2:
        Dv = spacetime.centered_1d_dx2(nv, dv, periodic=True).todense()
    elif dv_order == 4:
        Dv = spacetime.centered_1d_dx4(nv, dv, periodic=True).todense()

    # Initial condition
    X, V = np.meshgrid(xs, vs)
    A0 = 1 / (2 * np.pi) * np.exp(-0.5 * V[::-1] ** 2) * (1 + alpha * np.cos(k * X))
    A0 = A0.T

    # Define the problem
    ode = VlasovPoissonOde(Dx, Dv, vs, dx, dv)

    return ode, A0

#%% Two-stream instability
def make_two_stream(nx: int = 128, nv: int = 128, dx_order: int = 2, dv_order: int = 2):
    """
    Two-stream instability problem.

    Time interval: (0, 100)
    
    
    Parameters
    ----------
    nx : int, optional
        Number of spatial points, by default 128.
    nv : int, optional
        Number of velocity points, by default 128.
    dx_order : int, optional
        Order of the spatial derivative (2 or 4), by default 2.
    dv_order : int, optional
        Order of the velocity derivative (2 or 4), by default 2.

    Returns
    -------
    ode : VlasovPoissonOde
        Vlasov-Poisson ODE.
    A0 : ndarray
        Initial distribution function.    
    """
    # Constant variables
    v0 = 2.4
    k = 0.2
    alpha = 1e-3


    # Space field
    xmin = 0.0
    xmax = 10 * np.pi
    dx = (xmax - xmin) / nx
    xs = np.linspace(xmin, xmax, nx)
    if dx_order == 2:
        Dx = spacetime.centered_1d_dx2(nx, dx, periodic=True).todense()
    elif dx_order == 4:
        Dx = spacetime.centered_1d_dx4(nx, dx, periodic=True).todense()


    # Velocity field
    vmin = -6.0
    vmax = 6.0
    dv = (vmax - vmin) / nv
    vs = np.linspace(vmin, vmax, nv)
    if dv_order == 2:
        Dv = spacetime.centered_1d_dx2(nv, dv, periodic=True).todense()
    elif dv_order == 4:
        Dv = spacetime.centered_1d_dx4(nv, dv, periodic=True).todense()


    # Initial condition
    # Initial condition
    c = 1 / (2 * np.sqrt(2 * np.pi))
    X, V = np.meshgrid(xs, vs)
    A0 = c * ( np.exp(-0.5*(V[::-1] - v0)**2) + np.exp(-0.5*(V[::-1] + v0)**2) ) * (1 + alpha * np.cos(k * X))
    # Make it periodic
    A0[0, :] = A0[-2, :]
    A0[-1, :] = A0[1, :]
    A0[:, 0] = A0[:, -2]
    A0[:, -1] = A0[:, 1]
    A0 = A0.T

    # Define the problem
    ode = VlasovPoissonOde(Dx, Dv, vs, dx, dv)

    return ode, A0


