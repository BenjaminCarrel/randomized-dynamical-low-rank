# Randomized dynamical low-rank approximation

## Abstract

This paper introduces new dynamical low-rank (DLR) methods for solving large-scale matrix differential equations, motivated by algorithms from randomised numerical linear algebra.

The new methods consist essentially of two steps: a range estimation step followed by a post-processing step. The range estimation is done by a newly developed dynamical rangefinder method. Then, we propose two ways to perform the post-processing leading to two time-stepping methods that we respectively call dynamical randomised singular value decomposition (DRSVD) and dynamical generalised Nyström (DGN). The new methods have natural extensions to the rank-adaptive framework.

The cost of the new methods is comparable to existing DLR techniques, and their numerical performance make them very competitive; the experiments show that the new methods are very accurate and have little variance despite their inherent randomness. Our results suggest moreover that the new methods are robust to stiffness, and critical physical quantities such as the energy and geometry of the solution are numerically preserved. Interestingly, the new methods do not show any step-size restriction and can be used as one-step solvers. 

## Author

- Benjamin Carrel (University of Geneva)

## Reference

Link to arXiv: https://arxiv.org/abs/2410.17091


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
