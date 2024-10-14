"File for the experiments with the Allen-Cahn equation"

#%% Imports
import numpy as np
from low_rank_toolbox import SVD
from matrix_ode_toolbox.problems import make_allen_cahn
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import solve_adaptive_dlra

from params_plot import *
path = 'figures/allen-cahn/'
do_save = True

import os
if not os.path.exists(path):
    os.makedirs(path)


# %% Generate the problem
name = 'Allen-Cahn'
filename = 'allen_cahn'
n = 128
ode, X0 = make_allen_cahn(n)
t0, tf = 0, 10
nt = 20
t_span = (t0, tf)
ts = np.linspace(t0, tf, nt+1)

print(f'Problem generated: {name} with n={n}.')
print(f'Time span: {t_span} with {nt} time steps.')
print(ode)

#%% Reference solution
ref_solver = 'exponential_runge_kutta'
ref_solver_kwargs = {'order': 2, 'nb_substeps': 500}
ref_allen_cahn = solve_matrix_ivp(ode, t_span, X0, ref_solver, ref_solver_kwargs, t_eval=ts, monitor=True)
print('Done.')

#%% Define adaptive DLRA solvers

# Parameters and solvers
rtol = 1e-8
rangefinder_tol = 1e-12
atol = 1e-12
nb_substeps = 1
Y0 = SVD.truncated_svd(X0, rtol=rtol, atol=atol)

## Substep parameters
substep_kwargs = {'solver': 'exponential_runge_kutta', 'solver_kwargs': {'order': 2}, 'nb_substeps': 500}

# Solvers parameters
dlra_solvers = []
methods_kwargs = []
methods_name = []
methods_styles = []

## Unconventional
dlra_solvers += ['adaptive_unconventional']
methods_kwargs += [{'substep_kwargs': substep_kwargs}]
methods_name += ['Adaptive BUG']
methods_styles += ['-+']

## Dynamical randomized SVD
dlra_solvers += ['adaptive_dynamical_randomized_svd']
methods_kwargs += [{'substep_kwargs': substep_kwargs, 'rangefinder_tol': rangefinder_tol, 'do_ortho_sketching': False}]
methods_name += ['Adaptive DRSVD']
methods_styles += ['-x']

## Dynamical generalized Nystroem
dlra_solvers += ['adaptive_dynamical_generalized_nystroem']
methods_kwargs += [{'substep_kwargs': substep_kwargs, 'rangefinder_tol': rangefinder_tol, 'do_ortho_sketching': False}]
methods_name += ['Adaptive DGN']
methods_styles += ['-o']

#%% Solve the DLRAs
dlra_allen_cahn = []

for i, method in enumerate(dlra_solvers):
    print('****************')
    print(f'Solving with {methods_name[i]}...')
    dlra_allen_cahn.append(solve_adaptive_dlra(ode, t_span, Y0, method, solver_kwargs=methods_kwargs[i], rtol=rtol, atol=atol, t_eval=ts, monitor=True, nb_substeps=nb_substeps))
    print('Done.')

#%% Compute the errors
print('Computing the errors...')
dlra_relative_errors = []
for i, dlra_solution in enumerate(dlra_allen_cahn):
    errors = np.zeros(nt+1)
    for j, Xs in enumerate(ref_allen_cahn.Xs):
        errors[j] = np.linalg.norm(Xs - dlra_solution.Xs[j].todense(), ord='fro') / np.linalg.norm(Xs, ord='fro')
    dlra_relative_errors.append(errors)
best_rank_errors = np.zeros(nt+1)
for i, Xs in enumerate(ref_allen_cahn.Xs):
    best_rank_errors[i] = np.linalg.norm(Xs - SVD.truncated_svd(Xs, rtol=rtol, atol=atol).todense(), ord='fro') / np.linalg.norm(Xs, ord='fro')
print('Done.')

# Extract the rank over time
rank_over_time = np.zeros((len(dlra_solvers), nt+1))
for i, dlra_solution in enumerate(dlra_allen_cahn):
    for j, Xs in enumerate(dlra_solution.Xs):
        rank_over_time[i, j] = Xs.rank

true_rank_over_time = np.zeros(nt+1)
for j, Xs in enumerate(ref_allen_cahn.Xs):
    true_rank_over_time[j] = SVD.truncated_svd(Xs, rtol=rtol, atol=atol).rank

#%% Plot the results
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# Plot the relative error
for i, dlra_solution in enumerate(dlra_allen_cahn):
    axs[0].semilogy(ref_allen_cahn.ts, dlra_relative_errors[i], methods_styles[i], label=methods_name[i])
axs[0].plot(ref_allen_cahn.ts, best_rank_errors, label='Best approximation', linestyle='--', color='black')
# add tolerance
axs[0].axhline(y=rtol, color='black', linestyle=':', label='Tolerance')
axs[0].set_xlabel('Time', fontsize=20)
axs[0].set_ylabel('Relative error', fontsize=20)

# Plot the rank over time
for i, method_name in enumerate(methods_name):
    axs[1].plot(ref_allen_cahn.ts, rank_over_time[i], methods_styles[i], label=method_name)
axs[1].plot(ref_allen_cahn.ts, true_rank_over_time, '--', color='black', label='Reference')
axs[1].plot([], [], 'k:', label='Tolerance')
axs[1].set_xlabel('Time', fontsize=20)
axs[1].set_ylabel('Estimated rank', fontsize=20)
axs[1].legend(loc='best')

# Adjust layout
plt.tight_layout()

# Save the plots if required
if do_save:
    plt.savefig(f'{path}{filename}_relative_error_and_rank_over_time.pdf')

# Show the plots
plt.show()

# %%
