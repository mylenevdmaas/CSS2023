import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def conn_matrix_basic(n):
    """Returns nxn symmatric matrix for J with random numbers in [0,1]."""
    J_tri = np.tril(np.random.uniform(0, 1, size=(n, n)), -1)
    J = np.zeros((n,n)) + J_tri + J_tri.T
    return J

@njit
def random_spins(n):
    """Returns array of n spins in random configuration of -1 and 1."""
    values = np.random.randint(0, 2, size=n)
    values[values==0] = -1
    return values

@njit
def energy_diff(spins:np.array, c_matrix):
    """Computes energy difference for flipping one random spin based on J."""
    pos = np.random.randint(0, len(spins))
    delta_E = (spins[pos] * spins * c_matrix[pos]).sum() * 2
    return delta_E, pos

@njit
def metropolis(spins, n_iterations, T, c_matrix):
    """Runs one run of the metropolis algorithm with temperature T."""

    magnetisation_list = np.zeros(n_iterations)
    spins_timeseries = np.zeros((n_iterations, len(spins)))

    for i in range(n_iterations):
        diff, spin = energy_diff(spins, c_matrix)

        # accept if new energy is lower
        if diff <= 0:
            spins[spin] *=-1

        # else accept with probability based on Boltzman distribution
        else:
            if np.random.random() < np.exp(-diff/T):
                spins[spin] *=-1

        magnetisation_list[i] = abs(np.mean(spins))
        spins_timeseries[i] = spins

    # discard burn-in period and get mean of magnetization
    magnetisation_list = magnetisation_list[1000:]
    avg_magnetisation = np.mean(magnetisation_list) 
    spins_timeseries = spins_timeseries[1000:]
    
    # calculate susceptibility
    mean_of_squared = np.mean(magnetisation_list**2)
    susceptibility = (mean_of_squared - avg_magnetisation**2)/T

    return spins, avg_magnetisation, susceptibility, spins_timeseries

@njit
def multi_metropolis(n_simulations, n_iterations, T, n, c_matrix):
    """Runs n_simulations runs of the metropolis algorithm."""

    list_avg_magnetisation = np.zeros(n_simulations)
    list_sus = np.zeros(n_simulations)
    

    for i in range(n_simulations):

        # start with random spin config and J
        spins = random_spins(n)
        # c_matrix = conn_matrix_basic(n)

        # run metropolis
        _, list_avg_magnetisation[i], list_sus[i], spins_timeseries = metropolis(spins, n_iterations, T, c_matrix) 

    mean_magnet = np.mean(list_avg_magnetisation)
    std_magnet = np.std(list_avg_magnetisation)
    
    mean_sus = np.mean(list_sus)
    std_sus = np.std(list_sus)
    
    return mean_magnet, std_magnet, mean_sus, std_sus
    
@njit
def run_simulation(n_simulations:int, n_iterations:int, T_list:np.array, n:int, c_matrix=np.array):
    """Runs metropolis simulations for every temperature in a list of temperatures."""
    n_temp = len(T_list)

    means_mag = np.zeros(n_temp)
    stds_mag = np.zeros(n_temp)
    means_sus = np.zeros(n_temp)
    stds_sus = np.zeros(n_temp)
    for i, T in enumerate(T_list):
        means_mag[i], stds_mag[i], means_sus[i], stds_sus[i] = multi_metropolis(n_simulations, n_iterations, T, n, c_matrix)

    return [means_mag, stds_mag, means_sus, stds_sus]

@njit
def count_function(combo, coordinate, spins_timeseries, t, probabilities):
    """Counts the number of times a spin combination occurs in a timeseries of spin configurations.
    """
    [s_j, S_j, S_i] = combo
    [i,j] = coordinate
    [pSj, psj_Sj, pSj_Si, psj_Sj_Si] = probabilities

    if spins_timeseries[t-1][j] == S_j:
        pSj += 1

        if spins_timeseries[t-1][i] == S_i:
            pSj_Si += 1

        if spins_timeseries[t][j] == s_j:
            psj_Sj += 1

            if spins_timeseries[t-1][i] == S_i:
                psj_Sj_Si += 1
        
    return [pSj, psj_Sj, pSj_Si, psj_Sj_Si]

def conn_matrix_not_so_basic(n, fraction_of_zeros):
    """Returns nxn symmatric matrix for J with random numbers in [0,1]."""
    J_tri = np.tril(np.random.uniform(0, 1, size=(n, n)), -1)
    J = np.zeros((n,n)) + J_tri + J_tri.T

    f = int(np.floor(fraction_of_zeros*n*(n-1)/2))
    removed = []
    for i in range(f):
        while True:
            row = np.random.randint(0, n-1)
            cols = [num for num in range(0, n) if num != row]  # to not include central diagonal
            col = np.random.choice(cols)
            entry = [row, col]
            entry_T = [col, row]

            if entry in removed or entry_T in removed:
                continue
        
            J[row][col] = 0
            J[col][row] = 0

            removed.append(entry)
            break
    return J

@njit
def get_probability(coordinate, spins_timeseries, combos):
    """Calculates the probability of all possible spin combinations based on a timeseries of spin configurations."""
    count_array = np.zeros((8, 4))

    for t in range(1, len(spins_timeseries)):
        for k, combo in enumerate(combos):
            count_array[k] = count_function(combo, coordinate, spins_timeseries, t, count_array[k])

    return count_array
            

def plot_results(sim_data, T_list, sim_name, save=False):
    """Plots the results of a full simulation."""
    means_mag, stds_mag, means_sus, stds_sus = sim_data
    lower_bound = np.subtract(means_mag, stds_mag)
    upper_bound = np.add(means_mag, stds_mag)
    plt.plot(T_list, means_mag)
    plt.fill_between(T_list, lower_bound, upper_bound, alpha=0.3)
    plt.xlabel('T')
    plt.ylabel('M')
    plt.grid()

    if save:
        plt.savefig(f'{sim_name}_M.png', bbox_inches='tight')
    plt.show()    
    
    lower_bound = np.subtract(means_sus, stds_sus)
    upper_bound = np.add(means_sus, stds_sus)
    plt.plot(T_list, means_sus)
    plt.fill_between(T_list, lower_bound, upper_bound, alpha=0.3)
    plt.xlabel('T')
    plt.ylabel('Susceptibility')
    plt.grid()
    idx = np.argmax(sim_data[2])
    plt.vlines(T_list[idx],  min(sim_data[2]), max(sim_data[2])*1.2, linestyles='dashed', color = 'r')

    if save:
        plt.savefig(f'{sim_name}_sus.png', bbox_inches='tight')
    plt.show()

