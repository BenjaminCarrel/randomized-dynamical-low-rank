"File for the experiments on the two-stream instability"

#%% Imports
import numpy as np
from low_rank_toolbox import SVD
from matrix_ode_toolbox.problems import make_two_stream
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
name = 'Two-stream instability'
filename = 'two_stream_instability'
nx, nv = 128, 128 # original paper: 128, 128 and 256, 256 for the graphical abstract
ode, X0 = make_two_stream(nx, nv, dx_order=4, dv_order=4)
t0, tf = 0, 60
nt = tf*10
t_span = (t0, tf)
ts = np.linspace(t0, tf, nt+1)
print(f'Problem generated: {name} with nx={nx}, nv={nv}.')
print(ode)

#%% Reference solution
ref_solver = 'scipy'
ref_solver_kwargs = {'scipy_method': 'RK45', 'rtol': 1e-8, 'atol': 1e-8}
ref_two_stream = solve_matrix_ivp(ode, t_span, X0, ref_solver, ref_solver_kwargs, t_eval=ts, monitor=True)
print('Done.')

#%% Low-rank solvers

## Parameters and low-rank initial value
rank = 20 # original paper: 20, and 30 for the graphical abstract
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

## Augmented Unconventional
dlra_solvers += ['augmented_unconventional']
methods_kwargs += [{'nb_substeps': nb_substeps, 'substep_kwargs': substep_kwargs}]
methods_name += ['Augmented BUG']
methods_styles += ['-']

#%% Solve with the DLRA solvers
dlra_two_stream = []
for i, method in enumerate(dlra_solvers):
    print('****************')
    print(f'Solving with {methods_name[i]}...')
    dlra_two_stream.append(solve_dlra(ode, t_span, Y0, method, methods_kwargs[i], t_eval=ts, monitor=True))
    print('Done.')

#%% Visualize the results
times = [0, 30, 40, 60]

# Plot the results with 3 subplots
fig, axs = plt.subplots(len(dlra_two_stream)+1, len(times), figsize=(40, 10*(len(dlra_two_stream)+1)))
# plt.rcParams['font.size'] = 40
plt.grid(False)
# No tight layout
plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.1)
# Make title more readable
plt.subplots_adjust(top=0.9)
plt.subplots_adjust(bottom=0.1)
min_value = np.min(ref_two_stream.Xs[0])
max_value = np.max(ref_two_stream.Xs[0])
for i, time in enumerate(times):
    t_idx = np.argmin(np.abs(ref_two_stream.ts - time))
    axs[0, i].imshow(ref_two_stream.Xs[t_idx].T, vmin=min_value, vmax=max_value)
    axs[0, i].set_title(f'Reference at t={time}', fontsize=40)
    axs[0, i].set_xticks([])
    axs[0, i].set_yticks([])

    for j, dlra_solution in enumerate(dlra_two_stream):
        axs[j+1, i].imshow(dlra_solution.Xs[t_idx].todense().T, vmin=min_value, vmax=max_value)
        if methods_name[j] == 'Dynamical randomized SVD':
            axs[j+1, i].set_title(f'DRSVD at t={time}', fontsize=40)
        elif methods_name[j] == 'Dynamical generalized Nystroem':
            axs[j+1, i].set_title(f'DGN at t={time}', fontsize=40)
        else:
            axs[j+1, i].set_title(f'{methods_name[j]} at t={time}', fontsize=40)
        axs[j+1, i].set_xticks([])
        axs[j+1, i].set_yticks([])
    
if do_save:
    plt.savefig(f'{path}{filename}_solutions.pdf', bbox_inches='tight')
plt.show()

#%% Compute the physical quantities
print('Computing the electric energy...')
ref_electric_energy = np.zeros(nt+1)
best_rank_electric_energy = np.zeros(nt+1)
for i, Xs in enumerate(ref_two_stream.Xs):
    rho = ode.dv * np.sum(Xs, axis=1)
    ref_electric_energy[i] = ode.dx * np.linalg.norm(ode.electric_field(rho))
    best_rank_rho = ode.dv * np.sum(SVD.truncated_svd(Xs, rank).todense(), axis=1)
    best_rank_electric_energy[i] = ode.dx * np.linalg.norm(ode.electric_field(best_rank_rho))
dlra_electric_energy = []
for i, dlra_solution in enumerate(dlra_two_stream):
    electric_energy = np.zeros(nt+1)
    for j, Ys in enumerate(dlra_solution.Xs):
        rho = ode.dv * (Ys.U @ Ys.S @ np.sum(Ys.V.T.conj(), axis=1))
        electric_energy[j] = ode.dx * np.linalg.norm(ode.electric_field(rho))
    dlra_electric_energy.append(electric_energy)
print('Done.')

# Functions for mass, momentum and energy
def mass(A):
    return ode.dx * ode.dv * np.sum(A)

vs = np.linspace(-6, 6, 128)
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
for i, dlra_solution in enumerate(dlra_two_stream):
    dlra_mass.append(np.zeros(nt+1))
    dlra_momentum.append(np.zeros(nt+1))
    dlra_energy.append(np.zeros(nt+1))
for i, Xs in enumerate(ref_two_stream.Xs):
    ref_mass[i] = mass(Xs)
    ref_momentum[i] = momentum(Xs)
    ref_energy[i] = energy(Xs)
    for j, dlra_solution in enumerate(dlra_two_stream):
        Ys = dlra_solution.Xs[i]
        dlra_mass[j][i] = np.abs(mass(Ys.todense()) - ref_mass[i])
        dlra_momentum[j][i] = np.abs(momentum(Ys.todense()) - ref_momentum[i])
        dlra_energy[j][i] = np.abs(energy(Ys.todense()) - ref_energy[i])
print('Done.')

#%% Plot the results
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
axs[0, 0].semilogy(ts, ref_electric_energy, label='Reference', linestyle='-', color='black')
for i, dlra_solution in enumerate(dlra_two_stream):
    axs[0, 0].semilogy(ts, dlra_electric_energy[i], methods_styles[i], label=methods_name[i])
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Electric energy')

for i, dlra_solution in enumerate(dlra_two_stream):
    axs[0, 1].semilogy(ts, dlra_energy[i], methods_styles[i])
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Energy error')

for i, dlra_solution in enumerate(dlra_two_stream):
    axs[1, 0].semilogy(ts, dlra_mass[i], methods_styles[i])
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Mass error')

for i, dlra_solution in enumerate(dlra_two_stream):
    axs[1, 1].semilogy(ts, dlra_momentum[i], methods_styles[i])
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Momentum error')


# Legend and title
fig.legend(loc='upper left', bbox_to_anchor=(0.289, 0.73), fontsize=18)
# fig.suptitle(f'{name} - Rank {rank}', fontsize=20)
fig.tight_layout()
if do_save:
    fig.savefig(f'{path}{filename}_errors.pdf', bbox_inches='tight')

# %% Plot for the graphical abstract - subplots at final time with KSL, DRSVD and DGN and reference
if nx == 256 and nv == 256 and rank==30:
    fig, axs = plt.subplots(2, 2, figsize=(15, 20))
    axs[0, 0].imshow(ref_two_stream.Xs[-1].T, vmin=min_value, vmax=max_value)
    axs[0, 0].set_title(r'Reference (full rank)', fontsize=20)
    axs[0, 1].imshow(dlra_two_stream[0].Xs[-1].todense().T, vmin=min_value, vmax=max_value)
    axs[0, 1].set_title(r'Projector-splitting integrator (rank $30$)', fontsize=20)
    axs[1, 0].imshow(dlra_two_stream[1].Xs[-1].todense().T, vmin=min_value, vmax=max_value)
    axs[1, 0].set_title(r'Dynamical randomized SVD (rank $30$)', fontsize=20)
    axs[1, 1].imshow(dlra_two_stream[2].Xs[-1].todense().T, vmin=min_value, vmax=max_value)
    axs[1, 1].set_title(r'Dynamical generalized Nyström (rank $30$)', fontsize=20)

    # Remove ticks
    for i in range(2):
        for j in range(2):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    # Adjust layout to move titles below the figure
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.tight_layout()
    # fig.subplots_adjust(top=0.85)

    # Save the figure
    if do_save:
        fig.savefig(f'{path}{filename}_graphical_abstract.pdf', bbox_inches='tight')



# %%
