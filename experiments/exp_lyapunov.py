"File for the experiments with the Lyapunov equation"

#%% Imports
import numpy as np
from low_rank_toolbox import SVD
from matrix_ode_toolbox.problems import make_lyapunov_heat_square_dirichlet_kressner
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import solve_dlra
from tqdm import tqdm
from time import time, sleep

from params_plot import *
path = 'figures/lyapunov/'
do_save = True

import os
if not os.path.exists(path):
    os.makedirs(path)

#%% Set up the problem
print('Generating the problem...')
name = 'Lyapunov'
filename = 'lyapunov'
n = 256
ode, X0 = make_lyapunov_heat_square_dirichlet_kressner(n, skip_iv=True, alpha=1)
t0, tf = 0, 0.1
nt = 1
t_span = (t0, tf)
ts = np.linspace(t0, tf, nt+1)
print(f'Problem generated: {name} with n={n}.')
print(f'Time span: {t_span} with {nt} time steps.')
print(ode)

#%% Reference solution
ref_solver = 'exponential_runge_kutta'
ref_solver_kwargs = {'order': 2, 'nb_substeps': 1000}
# ref_solver = 'scipy'
# ref_solver_kwargs = {'scipy_method': 'RK45', 'atol': 1e-12, 'rtol': 1e-12}
ref_lyapunov = solve_matrix_ivp(ode, t_span, X0, ref_solver, ref_solver_kwargs, t_eval=ts, monitor=True)
print('Done.')

#%% DLRA methods

## Target rank and low-rank initial value
rank = 5
Y0 = SVD.truncated_svd(X0, rank)

## Substep parameters - reference solver
substep_kwargs = {'solver': 'exponential_runge_kutta', 'solver_kwargs': {'order': 2, 'nb_substeps': 1000}}
# substep_kwargs = {'solver': 'scipy', 'solver_kwargs': {'scipy_method': 'RK45', 'rtol': 1e-12, 'atol': 1e-12}}

# Solvers parameters
dlra_solvers = []
methods_kwargs = []
methods_name = []
methods_styles = []

## Projector-splitting
dlra_solvers += ['KSL']
methods_kwargs += [{'substep_kwargs': substep_kwargs, 'order': 1}]
methods_name += ['Projector-splitting']
methods_styles += ['-']

## PRK
dlra_solvers += ['PRK']
methods_kwargs += [{'order': 1}]
methods_name += ['PRK1']
methods_styles += ['-x']

## Randomized Runge-Kutta
dlra_solvers += ['randomized_runge_kutta']
methods_kwargs += [{'order': 1, 'rank': rank}]
methods_name += ['RRK1']
methods_styles += ['-o']

## PERK
dlra_solvers += ['PERK']
methods_kwargs += [{'order': 1}]
methods_name += ['PERK1']
methods_styles += ['-x']

## Unvonventional
dlra_solvers += ['unconventional']
methods_kwargs += [{'substep_kwargs': substep_kwargs}]
methods_name += ['BUG']
methods_styles += ['-+']

## Augmented Unvonventional
dlra_solvers += ['augmented_unconventional']
methods_kwargs += [{'substep_kwargs': substep_kwargs}]
methods_name += ['Augmented BUG']
methods_styles += ['-+']

#%% Solve with the DLRA solvers
dlra_lyapunov = []
for i, method in enumerate(dlra_solvers):
    print('****************')
    print(f'Solving with {methods_name[i]}...')
    dlra_lyapunov.append(solve_dlra(ode, t_span, Y0, method, methods_kwargs[i], t_eval=ts, monitor=True))
    print('Done.')

#%% Compute the errors
# Compute the errors
print('Computing the errors...')
dlra_relative_errors = []
for i, dlra_solution in enumerate(dlra_lyapunov):
    error = np.linalg.norm(ref_lyapunov.Xs[-1] - dlra_solution.Xs[-1].todense(), ord='fro') / np.linalg.norm(ref_lyapunov.Xs[-1], ord='fro')
    dlra_relative_errors.append(error)
best_rank_error = np.linalg.norm(ref_lyapunov.Xs[-1] - SVD.truncated_svd(ref_lyapunov.Xs[-1], rank).todense(), ord='fro') / np.linalg.norm(ref_lyapunov.Xs[-1], ord='fro')
print('Done.')

#%% Print the errors
print('Relative errors:')
for i, method in enumerate(dlra_solvers):
    print(f'{methods_name[i]}: {dlra_relative_errors[i]}')
print(f'Best rank error: {best_rank_error}')

# %% Relative error and oversampling

## Parameters
rank = 5
Y0 = SVD.truncated_svd(X0, rank)
list_oversampling = [0, 2, 5, 10]
list_oversamplings = [(p, p) for p in list_oversampling]
nb_repeat = 30

## Solve with the DLRA solvers
drsvd0_lyapunov = np.zeros((len(list_oversampling), nb_repeat), dtype=object)
drsvd1_lyapunov = np.zeros((len(list_oversampling), nb_repeat), dtype=object)
dgn0_lyapunov = np.zeros((len(list_oversampling), nb_repeat), dtype=object)
dgn1_lyapunov = np.zeros((len(list_oversampling), nb_repeat), dtype=object)
for i, oversampling in enumerate(list_oversampling):
    print('****************')
    print(f'Iteration {i+1}/{len(list_oversampling)}')
    for repeat in tqdm(range(nb_repeat)):
        seed = repeat
        
        # DRSVD (q=0)
        drsvd0_lyapunov[i, repeat] = solve_dlra(
            ode, t_span, Y0, 'dynamical_randomized_svd', 
            {'substep_kwargs': substep_kwargs, 'target_rank': rank, 'oversampling': oversampling, 'power_iterations': 0, 'do_ortho_sketching': False, 'seed': seed}, 
            t_eval=ts
        )
        # DRSVD (q=1)
        drsvd1_lyapunov[i, repeat] = solve_dlra(
            ode, t_span, Y0, 'dynamical_randomized_svd', 
            {'substep_kwargs': substep_kwargs, 'target_rank': rank, 'oversampling': oversampling, 'power_iterations': 1, 'do_ortho_sketching': False, 'seed': seed}, 
            t_eval=ts
        )

        # DGN (q=0)
        dgn0_lyapunov[i, repeat] = solve_dlra(
            ode, t_span, Y0, 'dynamical_generalized_nystroem', 
            {'substep_kwargs': substep_kwargs, 'target_rank': rank, 'oversampling_parameters': list_oversamplings[i], 'power_iterations': (0, 0), 'do_ortho_sketching': False, 'seed': seed}, 
            t_eval=ts
        )
        # DGN (q=1)
        dgn1_lyapunov[i, repeat] = solve_dlra(
            ode, t_span, Y0, 'dynamical_generalized_nystroem', 
            {'substep_kwargs': substep_kwargs, 'target_rank': rank, 'oversampling_parameters': list_oversamplings[i], 'power_iterations': (1, 1), 'do_ortho_sketching': False, 'seed': seed}, 
            t_eval=ts
        )

# %% Compute the mean errors at final time, and the first and third quartiles
print('Computing the errors...')
drsvd0_errors = np.zeros((len(list_oversampling),nb_repeat))
drsvd1_errors = np.zeros((len(list_oversampling),nb_repeat))
dgn0_errors = np.zeros((len(list_oversampling),nb_repeat))
dgn1_errors = np.zeros((len(list_oversampling),nb_repeat))
drsvd0_mean_errors = np.zeros((len(list_oversampling)))
drsvd1_mean_errors = np.zeros((len(list_oversampling)))
dgn0_mean_errors = np.zeros((len(list_oversampling)))
dgn1_mean_errors = np.zeros((len(list_oversampling)))
drsvd0_quartiles = np.zeros((len(list_oversampling), 2))
drsvd1_quartiles = np.zeros((len(list_oversampling), 2))
dgn0_quartiles = np.zeros((len(list_oversampling), 2))
dgn1_quartiles = np.zeros((len(list_oversampling), 2))


for i, oversampling in enumerate(list_oversampling):

    for repeat in range(nb_repeat):
        drsvd0_errors[i, repeat] = np.linalg.norm(ref_lyapunov.Xs[-1] - drsvd0_lyapunov[i, repeat].Xs[-1].todense(), ord='fro') / np.linalg.norm(ref_lyapunov.Xs[-1], ord='fro')
        drsvd1_errors[i, repeat] = np.linalg.norm(ref_lyapunov.Xs[-1] - drsvd1_lyapunov[i, repeat].Xs[-1].todense(), ord='fro') / np.linalg.norm(ref_lyapunov.Xs[-1], ord='fro')
        dgn0_errors[i, repeat] = np.linalg.norm(ref_lyapunov.Xs[-1] - dgn0_lyapunov[i, repeat].Xs[-1].todense(), ord='fro') / np.linalg.norm(ref_lyapunov.Xs[-1], ord='fro')
        dgn1_errors[i, repeat] = np.linalg.norm(ref_lyapunov.Xs[-1] - dgn1_lyapunov[i, repeat].Xs[-1].todense(), ord='fro') / np.linalg.norm(ref_lyapunov.Xs[-1], ord='fro')
    # Compute the mean and quartiles
    drsvd0_mean_errors[i] = np.mean(drsvd0_errors[i])
    drsvd1_mean_errors[i] = np.mean(drsvd1_errors[i])
    dgn0_mean_errors[i] = np.mean(dgn0_errors[i])
    dgn1_mean_errors[i] = np.mean(dgn1_errors[i])
    drsvd0_quartiles[i] = np.percentile(drsvd0_errors[i], [25, 75])
    drsvd1_quartiles[i] = np.percentile(drsvd1_errors[i], [25, 75])
    dgn0_quartiles[i] = np.percentile(dgn0_errors[i], [25, 75])
    dgn1_quartiles[i] = np.percentile(dgn1_errors[i], [25, 75])

print('Done.')

# %% Print the results

# Row 1 - Oversampling
print('Oversampling (p) &', end='')
for oversampling in list_oversampling:
    print(f' {oversampling} &', end='')
print('\\\\')

# Row 2 - DRSVD (q=0) (mean error)
print('DRSVD (q=0) &', end='')
for i in range(len(list_oversampling)):
    print(f' {drsvd0_mean_errors[i]:.2e} &', end='')
print('\\\\')

# Row 3 - DRSVD (q=0)
print('(quartiles) &', end='')
for i in range(len(list_oversampling)):
    print(f'({drsvd0_quartiles[i, 0]:.2e}, {drsvd0_quartiles[i, 1]:.2e}) &', end='')
print('\\\\')

# Row 4 - DRSVD (q=1) (mean error)
print('DRSVD (q=1) &', end='')
for i in range(len(list_oversampling)):
    print(f' {drsvd1_mean_errors[i]:.2e} &', end='')
print('\\\\')

# Row 5 - DRSVD (q=1)
print('(quartiles) &', end='')
for i in range(len(list_oversampling)):
    print(f'({drsvd1_quartiles[i, 0]:.2e}, {drsvd1_quartiles[i, 1]:.2e}) &', end='')
print('\\\\')

# Row 6 - DGN (q=0) (mean error)
print('DGN (q=0) &', end='')
for i in range(len(list_oversampling)):
    print(f' {dgn0_mean_errors[i]:.2e} &', end='')
print('\\\\')

# Row 7 - DGN (q=0)
print('(quartiles) &', end='')
for i in range(len(list_oversampling)):
    print(f'({dgn0_quartiles[i, 0]:.2e}, {dgn0_quartiles[i, 1]:.2e}) &', end='')
print('\\\\')

# Row 8 - DGN (q=1) (mean error)
print('DGN (q=1) &', end='')
for i in range(len(list_oversampling)):
    print(f' {dgn1_mean_errors[i]:.2e} &', end='')
print('\\\\')

# Row 9 - DGN (q=1)
print('(quartiles) &', end='')
for i in range(len(list_oversampling)):
    print(f'({dgn1_quartiles[i, 0]:.2e}, {dgn1_quartiles[i, 1]:.2e}) &', end='')
print('\\\\')

# %% Perform the method and compute the errors for several ranks
list_ranks = [2, 4, 8, 12, 16]
compute_time = np.zeros((4, len(list_ranks)))
errors = np.zeros((4, len(list_ranks)))

for i, rank in enumerate(list_ranks):
    Y0 = SVD.truncated_svd(X0, rank)

    # Pause for compilation
    print('****************')
    print(f'Rank {rank}')
    sleep(1)

    # Solve with augmented unconventional
    print('****************')
    t0 = time()
    augmented_bug = solve_dlra(ode, t_span, Y0, 'augmented_unconventional', {'substep_kwargs': substep_kwargs}, t_eval=ts, monitor=True)
    compute_time[0, i] = time() - t0

    # Solve with augmented unconventional
    print('****************')
    t0 = time()
    augmented_bug10 = solve_dlra(ode, t_span, Y0, 'augmented_unconventional', {'substep_kwargs': substep_kwargs, 'nb_substeps': 10}, t_eval=ts, monitor=True)
    compute_time[1, i] = time() - t0

    # Solve with dynamical randomized SVD
    print('****************')
    t0 = time()
    drsvd = solve_dlra(ode, t_span, Y0, 'dynamical_randomized_svd', {'substep_kwargs': substep_kwargs, 'target_rank': rank, 'oversampling': 5, 'power_iterations': 1, 'do_ortho_sketching': False}, t_eval=ts, monitor=True)
    compute_time[2, i] = time() - t0

    # Solve with dynamical generalized Nystroem
    print('****************')
    t0 = time()
    dgn = solve_dlra(ode, t_span, Y0, 'dynamical_generalized_nystroem', {'substep_kwargs': substep_kwargs, 'target_rank': rank, 'oversampling_parameters': (5, 5), 'do_ortho_gn': True, 'power_iterations': (1, 1), 'do_ortho_sketching': False}, t_eval=ts, monitor=True)
    compute_time[3, i] = time() - t0

    # Compute the errors
    errors[0, i] = np.linalg.norm(ref_lyapunov.Xs[-1] - augmented_bug.Xs[-1].todense(), ord='fro') / np.linalg.norm(ref_lyapunov.Xs[-1], ord='fro')
    errors[1, i] = np.linalg.norm(ref_lyapunov.Xs[-1] - augmented_bug10.Xs[-1].todense(), ord='fro') / np.linalg.norm(ref_lyapunov.Xs[-1], ord='fro')
    errors[2, i] = np.linalg.norm(ref_lyapunov.Xs[-1] - drsvd.Xs[-1].todense(), ord='fro') / np.linalg.norm(ref_lyapunov.Xs[-1], ord='fro')
    errors[3, i] = np.linalg.norm(ref_lyapunov.Xs[-1] - dgn.Xs[-1].todense(), ord='fro') / np.linalg.norm(ref_lyapunov.Xs[-1], ord='fro')


#%% Plot the results - Time as a function of the error
plt.figure(figsize=(12, 6))
plt.loglog(errors[0], compute_time[0], '+-', label=r'Augmented BUG ($h=0.1$)')
plt.loglog(errors[1], compute_time[1], '+-', label=r'Augmented BUG ($h=0.01$)')
plt.loglog(errors[2], compute_time[2], 'x-', label=r'DRSVD ($p=5, q=1, h=0.1$)')
plt.loglog(errors[3], compute_time[3], 'o-', label=r'DGN ($p=5, q=1, h=0.1$)')
for i, rank in enumerate(list_ranks):
    plt.text(errors[0, i] * 0.85, compute_time[0, i] * 0.85, rf'${rank}$', fontsize=18)
    plt.text(errors[1, i] * 0.85, compute_time[1, i] * 0.85, rf'${rank}$', fontsize=18)
    plt.text(errors[2, i] * 0.85, compute_time[2, i] * 0.85, rf'${rank}$', fontsize=18)
    plt.text(errors[3, i] * 0.85, compute_time[3, i] * 0.85, rf'${rank}$', fontsize=18)

plt.xlabel('Accuracy (relative error)')
plt.ylabel('Computation time (seconds)')
plt.gca().invert_xaxis()  # Reverse the x-axis
plt.ylim([0.5, 20])
# plt.legend(bbox_to_anchor=(1.0, 0.65), loc='upper left')
plt.legend()
if do_save:
    plt.savefig(f'{path}{filename}_computation_time.pdf')
plt.show()


# %%
