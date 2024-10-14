"File for the numerical experiments with the dynamical rangefinder"

#%% Imports
import numpy as np
from scipy.linalg import expm
from low_rank_toolbox import LowRankMatrix, SVD
from matrix_ode_toolbox import MatrixOde
from matrix_ode_toolbox.dlra import dynamical_rangefinder
from low_rank_toolbox.utils import randomized_rangefinder

from params_plot import *
path = 'figures/dynamical_rangefinder/'
do_save = True

import os
if not os.path.exists(path):
    os.makedirs(path)

#%% Set up the toy problem

# Parameters
np.random.seed(2810)
n = 100

# Random matrices
W1_tilde = np.random.rand(n, n)
W2_tilde = np.random.rand(n, n)
W1 = (W1_tilde - W1_tilde.T)/2
W2 = (W2_tilde - W2_tilde.T)/2

# Diagonal matrices
D1 = np.diag([10**(-i) for i in range(n)])

# ODE
def rhs(t, X):
    if isinstance(X, LowRankMatrix):
        X = X.todense()
    return W1.dot(X) + X + X.dot(W2.T)

problem1 = MatrixOde(D1, W1, W2)
problem1.ode_F = rhs

# Closed forms
def X1(t):
    return expm(t*W1).dot(np.exp(t)*D1).dot(expm(t*W2).T)
X10 = X1(0)


#%% Computations

# Parameters for the dynamical rangefinder
target_rank = 5
oversampling_parameters = np.arange(0, 13)
power_iterations = [0, 1]
nb_repeat = 100
h = 0.1
t_span = (0, h)

X1_ref = X1(h)

Y10 = SVD.from_dense(X10)

errors1_dr = np.zeros((len(oversampling_parameters), len(power_iterations), nb_repeat))
errors1_rf = np.zeros((len(oversampling_parameters), len(power_iterations), nb_repeat))

for i, p in enumerate(oversampling_parameters):
    for j, q in enumerate(power_iterations):
        print(f"p={p}, q={q}")
        for k in range(nb_repeat):
            # Run the dynamical rangefinder
            Q1_dr = dynamical_rangefinder(problem1, t_span, Y10, {'solver':'scipy'}, target_rank, p, q, do_ortho_sketch=False, seed=2810*k)
            # Compute the approximation and the error
            E1_dr = X1_ref - Q1_dr.dot(Q1_dr.T.dot(X1_ref))
            errors1_dr[i, j, k] = np.linalg.norm(E1_dr)/np.linalg.norm(X1_ref)

            # Run the rangefinder
            Q1_rf = randomized_rangefinder(X1_ref, target_rank, p, q, seed=2810*k)
            # Compute the approximation and the error
            E1_rf = X1_ref - Q1_rf.dot(Q1_rf.T.dot(X1_ref))
            errors1_rf[i, j, k] = np.linalg.norm(E1_rf)/np.linalg.norm(X1_ref)

# Best rank error
best_rank_error1 = np.linalg.norm(SVD.truncated_svd(X1_ref, target_rank).todense() - X1_ref) / np.linalg.norm(X1_ref)
best_rank_errors1 = np.zeros((len(oversampling_parameters)))
for i, p in enumerate(oversampling_parameters):
    best_rank_errors1[i] = np.linalg.norm(SVD.truncated_svd(X1_ref, target_rank + p).todense() - X1_ref) / np.linalg.norm(X1_ref)


#%% Plot the results
# Show a boxplot of the errors
fig, axs = plt.subplots(len(power_iterations), 2, sharey=True, figsize=(14, 7*len(power_iterations)))
for i, q in enumerate(power_iterations):
    boxprops = dict(linewidth=2)
    medianprops = dict(linewidth=2)
    whiskerprops = dict(linewidth=2)
    capprops = dict(linewidth=2)
    axs[i, 0].boxplot(errors1_dr[:, i, :].T, positions=[p for p in oversampling_parameters], showfliers=False, boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
    line1 = axs[i, 0].hlines(best_rank_error1, oversampling_parameters[0], oversampling_parameters[-1], colors='green', linestyles='dashed', label=r"Best rank-$r$ error")
    line2 = axs[i, 0].plot(oversampling_parameters, best_rank_errors1, 'k--', label=r"Best rank-$(r+p)$ error")
    line3 = axs[i, 0].hlines(1e-12, oversampling_parameters[0], oversampling_parameters[-1], colors='gray', linestyles='dashed', label="Solver precision")
    axs[i, 0].set_title(f"Dynamical rangefinder (q={q})", fontsize=24)
    axs[i, 0].set_yscale('log')
    axs[i, 0].set_xlabel(r"Oversampling parameter $p$", fontsize=20)
    axs[i, 0].set_ylabel("Relative error", fontsize=20)
    # Not all the labels are shown

    axs[i, 1].boxplot(errors1_rf[:, i, :].T, positions=[p for p in oversampling_parameters], showfliers=False, boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
    axs[i, 1].hlines(best_rank_error1, oversampling_parameters[0], oversampling_parameters[-1], colors='green', linestyles='dashed', label=r"Best rank-$r$ error")
    axs[i, 1].plot(oversampling_parameters, best_rank_errors1, 'k--', label=r"Best rank-$(r+p)$ error")
    line4 = axs[i, 1].hlines(1e-15, oversampling_parameters[0], oversampling_parameters[-1], colors='gray', linestyles='dotted', label="Machine precision")
    axs[i, 1].set_title(f"Rangefinder (q={q})", fontsize=24)
    axs[i, 1].set_yscale('log')
    axs[i, 1].set_xlabel(r"Oversampling parameter $p$", fontsize=20)

    # Legends
    axs[i, 0].legend(loc='lower left', fontsize=18)
    axs[i, 1].legend(loc='lower left', fontsize=18)


if do_save:
    plt.savefig("figures/dynamical_rangefinder/toy_problem.pdf")


plt.show()




# %%
