import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def conn_matrix_basic(n):
    J_tri = np.tril(np.random.uniform(0, 1, size=(n, n)), -1)
    J = np.zeros((n,n)) + J_tri + J_tri.T
    return J

@njit
def random_spins(n):
    values = np.random.randint(0, 2, size=n)
    values[values==0] = -1
    return values

@njit
def energy_diff(spins:np.array, c_matrix):
    pos = np.random.randint(0, len(spins))
    delta_E = (spins[pos] * spins * c_matrix[pos]).sum() * 2
    return delta_E, pos

@njit
def metropolis(spins, n_iterations, T, c_matrix):

    magnetisation_list = np.zeros(n_iterations)

    for i in range(n_iterations):
        diff, spin = energy_diff(spins, c_matrix)
        if diff <= 0:
            spins[spin] *=-1
        else:
            if np.random.random() < np.exp(-diff/T):
                spins[spin] *=-1

        magnetisation_list[i] = abs(np.mean(spins))

    magnetisation_list = magnetisation_list[500:]
    avg_magnetisation = np.mean(magnetisation_list) 
    
    mean_of_squared = np.mean(magnetisation_list**2)
    susceptibility = (mean_of_squared - avg_magnetisation**2)/T

    return spins, avg_magnetisation, susceptibility

@njit
def multi_metropolis(n_simulations, n_iterations, T, n):

    list_avg_magnetisation = np.zeros(n_simulations)
    list_sus = np.zeros(n_simulations)

    for i in range(n_simulations):
        spins = random_spins(n)
        c_matrix = conn_matrix_basic(n)
        _, list_avg_magnetisation[i], list_sus[i] = metropolis(spins, n_iterations, T, c_matrix) 

    mean_magnet = np.mean(list_avg_magnetisation)
    std_magnet = np.std(list_avg_magnetisation)
    
    mean_sus = np.mean(list_sus)
    std_sus = np.std(list_sus)
    
    return mean_magnet, std_magnet, mean_sus, std_sus
    
@njit
def run_simulation(n_simulations:int, n_iterations:int, T_list:np.array, n:int)

    n_temp = len(T_list)

    means_mag = np.zeros(n_temp)
    stds_mag = np.zeros(n_temp)
    means_sus = np.zeros(n_temp)
    stds_sus = np.zeros(n_temp)
    for i, T in enumerate(T_list):
        means_mag[i], stds_mag[i], means_sus[i], stds_sus[i] = multi_metropolis(n_simulations, n_iterations, T, n)

    return [means_mag, stds_mag, means_sus, stds_sus]

def plot_results(sim_data, T_list, sim_name, save=False):
    means_mag, stds_mag, means_sus, stds_sus = sim_data
    lower_bound = np.subtract(means_mag, stds_mag)
    upper_bound = np.add(means_mag, stds_mag)
    plt.plot(T_list, means_mag)
    plt.fill_between(T_list, lower_bound, upper_bound, alpha=0.3)
    plt.xlabel('T')
    plt.ylabel('M')
    plt.grid()

    if save:
        plt.savefig(f'output/{sim_name}_M.png', bbox_inches='tight')
    plt.show()
    
    lower_bound = np.subtract(means_sus, stds_sus)
    upper_bound = np.add(means_sus, stds_sus)
    plt.plot(T_list, means_sus)
    plt.fill_between(T_list, lower_bound, upper_bound, alpha=0.3)
    plt.xlabel('T')
    plt.ylabel('Susceptibility')
    plt.grid()

    if save:
        plt.savefig(f'output/{sim_name}_sus.png', bbox_inches='tight')
    plt.show()