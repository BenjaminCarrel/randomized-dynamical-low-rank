"File for the experiments with the Burgers equation"

#%% Imports
import numpy as np
from low_rank_toolbox import SVD
from matrix_ode_toolbox.problems import make_stochastic_Burgers
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import solve_dlra

from params_plot import *
path = 'figures/burgers/'
do_save = True

import os
if not os.path.exists(path):
    os.makedirs(path)


#%% Generate the problem
name = 'Stochastic Burgers'
filename = 'stochastic_burgers'
nx = 256
s = 64
ode, X0 = make_stochastic_Burgers(nx, s, nu=0.01, sigma=0.001, d=4)
t0, tf = 0, 0.2
t_span = (t0, tf)
nt = 20
ts = np.linspace(t0, tf, nt+1)

print(f'Problem generated: {name} with nx={nx}, s={s}')
print(f'Time span: {t_span} with {nt} time steps.')
print(ode)


#%% Reference solution
ref_solver = 'scipy'
ref_solver_kwargs = {}
ref_burgers = solve_matrix_ivp(ode, t_span, X0, ref_solver, ref_solver_kwargs, t_eval=ts, monitor=True)
print('Done.')

# Define DLRA solvers
#%% Parameters and solvers
rank = 10
Y0 = SVD.truncated_svd(X0, rank)
nb_substeps = 1
p = 10
q = 0

## Substep parameters
substep_kwargs = {'solver': 'scipy', 'solver_kwargs': {'scipy_method': 'RK45', 'rtol': 1e-12, 'atol': 1e-12}}

# Solvers parameters
dlra_solvers = []
methods_kwargs = []
methods_name = []
methods_styles = []


## Projector-splitting
dlra_solvers += ['KSL']
methods_kwargs += [{'substep_kwargs': substep_kwargs, 'order': 1, 'nb_substeps': nb_substeps}]
methods_name += ['KSL']
methods_styles += ['-v']

## Unconventional
dlra_solvers += ['unconventional']
methods_kwargs += [{'nb_substeps': nb_substeps, 'substep_kwargs': substep_kwargs}]
methods_name += ['BUG']
methods_styles += ['-^']

## Augmeneted unconventional
dlra_solvers += ['augmented_unconventional']
methods_kwargs += [{'nb_substeps': nb_substeps, 'substep_kwargs': substep_kwargs}]
methods_name += ['Augmented BUG']
methods_styles += ['-+']

## Dynamical randomized SVD
dlra_solvers += ['dynamical_randomized_svd']
methods_kwargs += [{'nb_substeps': nb_substeps, 'substep_kwargs': substep_kwargs, 'target_rank': rank, 'oversampling':p, 'power_iterations': q, 'do_ortho_sketching': False}]
methods_name += [f'DRSVD (p={p}, q={q})']
methods_styles += ['-x']

## Dynamical generalized Nystroem
dlra_solvers += ['dynamical_generalized_nystroem']
methods_kwargs += [{'nb_substeps': nb_substeps, 'substep_kwargs': substep_kwargs, 'target_rank': rank, 'oversampling_parameters': (p, p), 'power_iterations': (q, q), 'do_ortho_sketching': False}]
methods_name += [f'DGN (p={p}, q={q})']
methods_styles += ['-o']


# Solve with the DLRA solvers
dlra_burgers = []
for i, method in enumerate(dlra_solvers):
    print('****************')
    print(f'Solving with {methods_name[i]}...')
    dlra_burgers.append(solve_dlra(ode, t_span, Y0, method, methods_kwargs[i], t_eval=ts, monitor=True))
    print('Done.')


# Compute the errors
errors = []
for i, dlra_burger in enumerate(dlra_burgers):
    errors.append([np.linalg.norm(ref_burgers.Xs[j] - dlra_burger.Xs[j].todense(), 'fro')/np.linalg.norm(ref_burgers.Xs[j], 'fro') for j in range(len(ts))])
errors.append([np.linalg.norm(ref_burgers.Xs[j] - SVD.truncated_svd(ref_burgers.Xs[j],rank).todense(), 'fro')/np.linalg.norm(ref_burgers.Xs[j], 'fro') for j in range(len(ts))])

# Plot the results
plt.figure(figsize=(12, 6))
for i, error in enumerate(errors[:-1]):
    plt.semilogy(ts, error, methods_styles[i], label=methods_name[i])
plt.semilogy(ts, errors[-1], label=f'Best rank-{rank} approximation', linestyle='--', color='black')
plt.xlabel('Time')
plt.ylabel('Relative error')
plt.legend(bbox_to_anchor=(1, 0.75), loc='upper left')
if do_save:
    plt.savefig(path + filename + '_relative_error.pdf', bbox_inches='tight')
plt.show()
# %%
