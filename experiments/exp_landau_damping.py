"File for the experiments on the Landau damping"

#%% Imports
import numpy as np
from low_rank_toolbox import SVD
from matrix_ode_toolbox.problems import make_landau_damping
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import solve_dlra

from params_plot import *

# Parameters for saving the plots
path = 'figures/vlasov-poisson/'
do_save = True

# Make directory if it does not exist
import os
if do_save:
    if not os.path.exists(path):
        os.makedirs(path)

#%% Generate the problem
name = 'Landau damping'
filename = 'landau_damping'
nx, nv = 64, 256 # original paper: 64, 256
ode, X0 = make_landau_damping(nx, nv, dx_order=4, dv_order=4)
t0, tf = 0, 40
nt = 1000
t_span = (t0, tf)
ts = np.linspace(t0, tf, nt+1)
print(f'Problem generated: {name} with nx={nx}, nv={nv}.')
print(ode)

#%% Reference solution
ref_solver = 'scipy'
ref_solver_kwargs = {'scipy_method': 'RK45', 'rtol': 1e-8, 'atol': 1e-8}
ref_landau_damping = solve_matrix_ivp(ode, t_span, X0, ref_solver, ref_solver_kwargs, t_eval=ts, monitor=True)
print('Done.')

#%% Low-rank solvers

## Parameters and low-rank initial value
rank = 10
Y0 = SVD.truncated_svd(X0, rank)
dt = 0.1
if dt > 0.1:
    raise ValueError('dt should be smaller or equal to 1.')
nb_substeps = int((tf-t0)/nt/dt)
nb_substeps = 1 if nb_substeps == 0 else nb_substeps

## Substep parameters
substep_kwargs = {'solver': 'scipy', 'solver_kwargs': {'scipy_method': 'RK45', 'rtol': 1e-8, 'atol': 1e-8}}

# Solvers parameters
dlra_solvers = []
methods_kwargs = []
methods_name = []
methods_styles = []

## Projector-splitting
dlra_solvers += ['KSL']
methods_kwargs += [{'substep_kwargs': substep_kwargs, 'order': 2, 'nb_substeps': nb_substeps}]
methods_name += ['KSL2']
methods_styles += ['-']

## Dynamical randomized SVD
dlra_solvers += ['dynamical_randomized_svd']
methods_kwargs += [{'nb_substeps': nb_substeps, 'substep_kwargs': substep_kwargs, 'target_rank': rank, 'oversampling': 5, 'power_iterations': 1, 'do_ortho_sketching': False}]
methods_name += ['DRSVD']
methods_styles += ['-']

## Dynamical generalized Nystroem
dlra_solvers += ['dynamical_generalized_nystroem']
methods_kwargs += [{'nb_substeps': nb_substeps, 'substep_kwargs': substep_kwargs, 'target_rank': rank, 'oversampling_parameters': (5, 5), 'power_iterations': (1, 1), 'do_ortho_sketching': False}]
methods_name += ['DGN']
methods_styles += ['-']

#%% Solve with the DLRA solvers
dlra_landau_damping = []
for i, method in enumerate(dlra_solvers):
    print('****************')
    print(f'Solving with {methods_name[i]}...')
    dlra_landau_damping.append(solve_dlra(ode, t_span, Y0, method, methods_kwargs[i], t_eval=ts, monitor=True))
    print('Done.')

#%% Compute the electric energy
print('Computing the electric energy...')
ref_electric_energy = np.zeros(nt+1)
best_rank_electric_energy = np.zeros(nt+1)
for i, Xs in enumerate(ref_landau_damping.Xs):
    rho = ode.dv * np.sum(Xs, axis=1)
    ref_electric_energy[i] = ode.dx * np.linalg.norm(ode.electric_field(rho))
    best_rank_rho = ode.dv * np.sum(SVD.truncated_svd(Xs, rank).todense(), axis=1)
    best_rank_electric_energy[i] = ode.dx * np.linalg.norm(ode.electric_field(best_rank_rho))
dlra_electric_energy = []
for i, dlra_solution in enumerate(dlra_landau_damping):
    electric_energy = np.zeros(nt+1)
    for j, Ys in enumerate(dlra_solution.Xs):
        rho = ode.dv * (Ys.U @ Ys.S @ np.sum(Ys.V.T.conj(), axis=1))
        electric_energy[j] = ode.dx * np.linalg.norm(ode.electric_field(rho))
    dlra_electric_energy.append(electric_energy)
theoretical_electric_energy = np.zeros(nt+1)
decay_rate = 0.153
for i, t in enumerate(ref_landau_damping.ts):
    theoretical_electric_energy[i] = ref_electric_energy[0] * (1-decay_rate)**(2*t)
print('Done.')

#%% Compute the other quantities
# Functions for mass, momentum and energy
def mass(A):
    return ode.dx * ode.dv * np.sum(A)

vs = np.linspace(-6, 6, nv)
def momentum(A):
    return ode.dx * ode.dv * np.sum(vs.T * A)

def energy(A):
    q1 = ode.dx * ode.dv * np.sum((vs.T**2) * A)
    rho = ode.dv * np.sum(A, axis=1)
    E = ode.electric_field(rho)
    q2 = ode.dx * np.sum(E)**2
    return 0.5 * q1 + 0.5 * q2


# Compute the mass, momentum and energy errors
print('Computing the mass, momentum and energy...')
ref_mass = np.zeros(nt+1)
ref_momentum = np.zeros(nt+1)
ref_energy = np.zeros(nt+1)
dlra_mass = []
dlra_momentum = []
dlra_energy = []
for i, dlra_solution in enumerate(dlra_landau_damping):
    dlra_mass.append(np.zeros(nt+1))
    dlra_momentum.append(np.zeros(nt+1))
    dlra_energy.append(np.zeros(nt+1))
for i, Xs in enumerate(ref_landau_damping.Xs):
    ref_mass[i] = mass(Xs)
    ref_momentum[i] = momentum(Xs)
    ref_energy[i] = energy(Xs)
    for j, dlra_solution in enumerate(dlra_landau_damping):
        Ys = dlra_solution.Xs[i]
        dlra_mass[j][i] = np.abs(mass(Ys.todense()) - ref_mass[i])
        dlra_momentum[j][i] = np.abs(momentum(Ys.todense()) - ref_momentum[i])
        dlra_energy[j][i] = np.abs(energy(Ys.todense()) - ref_energy[i])
print('Done.')

#%% Plot the electric energy, mass, momentum and energy
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
axs[0, 0].semilogy(ref_landau_damping.ts, ref_electric_energy, label='Reference', linestyle='-', color='black')
axs[0, 0].semilogy(ref_landau_damping.ts, theoretical_electric_energy, label='Analytic decay rate', linestyle='-.', color='red')
for i, dlra_solution in enumerate(dlra_landau_damping):
    axs[0, 0].semilogy(ref_landau_damping.ts, dlra_electric_energy[i], methods_styles[i], label=methods_name[i])
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Electric energy')
axs[0, 0].set_ylim([1e-10, 1e-1])
# axs[0, 0].grid()
# axs[0, 0].set_title(f'{name} - Electric energy - Rank {rank}')
# axs[0, 0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

for i, dlra_solution in enumerate(dlra_landau_damping):
    axs[0, 1].semilogy(ref_landau_damping.ts, dlra_energy[i], methods_styles[i])
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Energy error')
axs[0, 1].set_ylim([1e-10, 1e-1])
# axs[0, 1].grid()
# axs[1, 1].set_title(f'{name} - Energy error - Rank {rank}')
# axs[1, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

for i, dlra_solution in enumerate(dlra_landau_damping):
    axs[1, 0].semilogy(ref_landau_damping.ts, dlra_mass[i], methods_styles[i])
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Mass error')
axs[1, 0].set_ylim([1e-15, 1e-9])
# axs[1, 0].grid()
# axs[0, 1].set_title(f'{name} - Mass error - Rank {rank}')
# axs[0, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

for i, dlra_solution in enumerate(dlra_landau_damping):
    axs[1, 1].semilogy(ref_landau_damping.ts, dlra_momentum[i], methods_styles[i])
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Momentum error')
axs[1, 1].set_ylim([1e-15, 1e-9])
# axs[1, 1].grid()
# axs[1, 0].set_title(f'{name} - Momentum error - Rank {rank}')
# axs[1, 0].legend(loc='center left', bbox_to_anchor=(1, 0.5))


# Legend and title
fig.legend(loc='upper left', bbox_to_anchor=(0.58, 0.98), fontsize=18)
# fig.suptitle(f'Linear {name} - Rank {rank}', fontsize=20)

if do_save:
    fig.savefig(f'{path}{filename}_errors.pdf', bbox_inches='tight')


# %%
