"File for setting up the parameters for the plots"

# Import matplotlib
import matplotlib.pyplot as plt

# Custom parameters
plt.rcParams['font.size'] = 16
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['figure.autolayout'] = True

# Save custom parameters
custom_params = plt.rcParams.copy()