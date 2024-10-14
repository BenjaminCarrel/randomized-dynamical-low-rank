# Randomized dynamical low-rank approximation

## Abstract

We introduce new dynamical low-rank (DLR) methods for solving large-scale matrix differential equations, motivated by algorithms from randomized numerical linear algebra.
The new methods consist essentially of two steps: a range estimation step followed by a post-processing step.
For the range estimation, we developed a method that we call dynamical rangefinder.
Then, we propose two ways to perform the post-processing, leading to two time-stepping methods that we respectively call dynamical randomized singular value decomposition (DRSVD) and dynamical generalized Nyström (DGN).
The cost of the new techniques is comparable to existing DLR techniques.
% Under certain conditions, the two new time-stepping methods are exact on low-rank matrices.
Numerical experiments show that the new methods are robust to stiffness, structure preserving, and are very accurate on various problems.
Interestingly, the new methods do not show any step-size restriction and can therefore be used as one-step solvers.
For each method, a rank-adaptive variant is also proposed.

## Author

- Benjamin Carrel (University of Geneva)

## Reference

[Randomized methods for the dynamical low-rank approximation]()

Please use the following template for citing:
'''
TO BE UPDATED
'''



## Installation instructions

Download or clone the folder with the command
'''
git clone https://github.com/BenjaminCarrel/randomized-dynamical-low-rank.git
cd randomized-dynamical-low-rank
'''

### General

Before running the experiments, you will need to install the following libraries:
- python (3.12)
- pip (24.2)
- numpy (2.1.2)
- scipy (1.14.1)
- matplotlib (3.9.2)
- jupyter (1.1.1)
- tqdm (4.66.5)
And add the folders `low_rank_toolbox` and `matrix_ode_toolbox` as source.

### Conda installation

In the terminal, go to the folder and create a new conda environment with the command

`conda env create --file environment.yml`

Then, activate the environment with

`conda activate randomized-dlra`

Finally, compile and install the package with

`pip install --compile .`
