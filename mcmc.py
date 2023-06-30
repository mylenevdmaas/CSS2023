import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def conn_matrix_basic(n:int):
    """Returns nxn symmatric matrix for J with random numbers in [0,1]."""
    J_tri = np.tril(np.random.uniform(0, 1, size=(n, n)), -1)
    J = np.zeros((n,n)) + J_tri + J_tri.T
    return J

@njit
def mean_matrix(n:int, mean:float):
    J_tri = np.tril(np.random.uniform(mean - 0.1, mean + 0.1, size=(n, n)), -1)
    J = np.zeros((n,n)) + J_tri + J_tri.T
    return J
    
@njit
def conn_matrix_norm(n:int):
    """Returns nxn symmatric matrix for J with random numbers in [0,1] from a normal distribution."""
    J_tri = np.tril(np.random.normal(0.5, 0.5, size = (n,n)), -1)
    J = np.zeros((n,n)) + J_tri + J_tri.T
    J -=  np.min(J)
    J /= np.max(J)
    return J

@njit
def conn_matrix_power(n:int):
    """Returns nxn symmatric matrix for J with random numbers in [0,1] from a powerlaw 
    distribution with exponent 1.4. """
    J_tri = np.tril((1-np.random.power(2.4, size = (n,n))), -1)
    J = np.zeros((n,n)) + J_tri + J_tri.T
    return J 

@njit
def conn_matrix_fraction_zeros(n:int, fraction:float):
    """Returns nxn symmatric matrix for J with random numbers in [0,1]."""
    J_tri = np.tril(np.random.uniform(0, 1, size=(n, n)), -1)

    # set fraction of matrix to zero
    destruction = np.random.uniform(0,1 , size = (n, n))
    for i in range(len(J_tri)):
        J_tri[i][destruction[i] > 1 - fraction] = 0

    J = np.zeros((n,n)) + J_tri + J_tri.T
    return J

# c_matrix_fun_dict = {
#     'basic': conn_matrix_basic,
#     'mean': mean_matrix,
#     'norm': conn_matrix_norm,
#     'power': conn_matrix_power,
#     'zeros': conn_matrix_fraction_zeros
# }


@njit
def random_spins(n:int):
    """Returns array of n spins in random configuration of -1 and 1."""
    values = np.random.randint(0, 2, size=n)
    values[values==0] = -1
    return values

@njit
def energy_diff(spins:np.array, c_matrix:np.array):
    """Computes energy difference for flipping one random spin based on J."""
    pos = np.random.randint(0, len(spins))
    delta_E = (spins[pos] * spins * c_matrix[pos]).sum() * 2
    return delta_E, pos

@njit
def metropolis(spins:np.array, n_iterations:int, T:float, c_matrix:np.array, burn_in:int=1000):
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
    magnetisation_list = magnetisation_list[burn_in:]
    avg_magnetisation = np.mean(magnetisation_list) 
    spins_timeseries = spins_timeseries[burn_in:]
    
    # calculate susceptibility
    mean_of_squared = np.mean(magnetisation_list**2)
    susceptibility = (mean_of_squared - avg_magnetisation**2)/T

    return spins, avg_magnetisation, susceptibility, spins_timeseries

@njit
def multi_metropolis_diff_connectivity(n_simulations:int, n_iterations:int, T:float, n:int, c_matrix_fun=mean_matrix, c_matrix_arg:float=0.5, burn_in:int=1000):
    """Runs n_simulations runs of the metropolis algorithm."""

    # c_matrix_fun = c_matrix_fun_dict[c_matrix_fun]

    list_avg_magnetisation = np.zeros(n_simulations)
    list_sus = np.zeros(n_simulations)

    for i in range(n_simulations):

        # start with random spin config and J
        spins = random_spins(n)
        c_matrix = c_matrix_fun(n, c_matrix_arg)

        # run metropolis
        _, list_avg_magnetisation[i], list_sus[i], _ = metropolis(spins, n_iterations, T, c_matrix, burn_in) 

    mean_magnet = np.mean(list_avg_magnetisation)
    std_magnet = np.std(list_avg_magnetisation)
    
    mean_sus = np.mean(list_sus)
    std_sus = np.std(list_sus)
    
    return mean_magnet, std_magnet, mean_sus, std_sus

@njit
def multi_metropolis_basic(n_simulations:int, n_iterations:int, T:float, n:int, c_matrix:np.array=None, c_matrix_fun=conn_matrix_basic, burn_in:int=1000):
    """Runs n_simulations runs of the metropolis algorithm."""

    # c_matrix_fun = c_matrix_fun_dict[c_matrix_fun]

    list_avg_magnetisation = np.zeros(n_simulations)
    list_sus = np.zeros(n_simulations)

    for i in range(n_simulations):

        # start with random spin config and J
        spins = random_spins(n)
        if c_matrix != None:
            c_matrix = c_matrix_fun(n)

        # run metropolis
        _, list_avg_magnetisation[i], list_sus[i], _ = metropolis(spins, n_iterations, T, c_matrix, burn_in) 

    mean_magnet = np.mean(list_avg_magnetisation)
    std_magnet = np.std(list_avg_magnetisation)
    
    mean_sus = np.mean(list_sus)
    std_sus = np.std(list_sus)
    
    return mean_magnet, std_magnet, mean_sus, std_sus
    
@njit
def run_simulation(n_simulations:int, n_iterations:int, T_list:np.array, n:int, c_matrix:np.array=None, c_matrix_fun=conn_matrix_basic, burn_in:int=1000):
    """Runs metropolis simulations for every temperature in a list of temperatures."""
    n_temp = len(T_list)

    means_mag = np.zeros(n_temp)
    stds_mag = np.zeros(n_temp)
    means_sus = np.zeros(n_temp)
    stds_sus = np.zeros(n_temp)
    for i, T in enumerate(T_list):
        means_mag[i], stds_mag[i], means_sus[i], stds_sus[i] = multi_metropolis_basic(n_simulations, n_iterations, T, n, c_matrix, c_matrix_fun, burn_in)

    return [means_mag, stds_mag, means_sus, stds_sus]

@njit
def get_probability(coordinate, spins_timeseries):
    """Computes counts for each combination of values in timeseries."""

    # get coordinates
    i,j = coordinate

    # get "current" spin orientations for j
    s_j = spins_timeseries[1:, j]

    # get "past" spin orientations for i and j
    S_j = spins_timeseries[:-1, j]
    S_i = spins_timeseries[:-1, i]

    count_array = np.zeros((8,4))

    # compute pSj, psj_Sj, pSj_Si, psj_Sj_Si for all 8 combinations
    i = 0
    for j_1 in [-1,1]:
        pSj = (S_j==j_1).sum()
        for j_0 in [-1,1]:
            psj_Sj = ((s_j==j_0) & (S_j==j_1)).sum()
            for i_1 in [-1,1]:
                pSj_Si = ((S_i==i_1) & (S_j==j_1)).sum()
                psj_Sj_Si = ((s_j==j_0) & (S_j==j_1) & (S_i==i_1)).sum()
                count_array[i] = np.array([pSj, psj_Sj, pSj_Si, psj_Sj_Si])
                i += 1

    return count_array

@njit
def TE(spins_timeseries, c_matrix):
    """Computes transfer entropy for a Metropolis timeseries."""

    c_total = 0

    # loop through all nonzero connections
    for coordinate in np.stack(c_matrix.nonzero(), axis=1):

        # get counts for all 8 possible combinations
        count_array = get_probability(coordinate, spins_timeseries)

        # remove any row with a zero value
        count_array = count_array[(count_array[:,3] != 0) & (count_array[:,2] != 0) & (count_array[:,1] != 0) & (count_array[:,0] != 0)]
        
        # convert to probabilities
        prob = (count_array / (spins_timeseries.shape[0]-1)).T

        # compute Cij
        c = np.sum(prob[3] * np.log((prob[1] * prob[2]) / (prob[3] * prob[0])))
        c_total += c

    return c_total

@njit
def mutual_info(coordinate, spins_timeseries):
    """Computes mutual information based on time series for node x and y: I(x;y)."""

    # get time series
    x,y = coordinate
    x_list = spins_timeseries[:,x]
    y_list = spins_timeseries[:,y]

    count_array = np.zeros((4,3))
    i = 0

    # count occurences for each of 4 possible states
    for x in [-1,1]:
        px = (x_list==x).sum()
        for y in [-1,1]:
            py = (y_list==y).sum()
            pxy = ((x_list==x) & (y_list==y)).sum()
            count_array[i] = np.array([px, py, pxy])
            i += 1
    
    # delete all rows with zero and convert to probabilities
    count_array = count_array[(count_array[:,2] != 0) & (count_array[:,1] != 0) & (count_array[:,0] != 0)]
    prob = (count_array / len(spins_timeseries)).T

    # compute mutual information
    m_i = np.sum(prob[2] * np.log(prob[2] / (prob[0] * prob[1])))

    return m_i

@njit
def con_mutual_info(coordinate, spins_timeseries):
    """Computes conditional mutual information based on time series with z as condition: I(x;y|z)"""

    # get time series
    x, y, z = coordinate
    x_list = spins_timeseries[:,x]
    y_list = spins_timeseries[:,y]
    z_list = spins_timeseries[:,z]

    count_array = np.zeros((8,4))
    i = 0

    # count occurences for all 8 possible states
    for z in [-1,1]:
        pz = (z_list==z).sum()
        for x in [-1,1]:
            pxz = ((x_list==x) & (z_list==z)).sum()
            for y in [-1,1]:
                pyz = ((y_list==y) & (z_list==z)).sum()
                pxyz = ((x_list==x) & (y_list==y) & (z_list==z)).sum()
                count_array[i] = np.array([pz, pxz, pyz, pxyz])
                i += 1

    # remove any row with a zero value
    count_array = count_array[(count_array[:,3] != 0) & (count_array[:,2] != 0) & (count_array[:,1] != 0) & (count_array[:,0] != 0)]
        
    # convert to probabilities
    prob = (count_array / (spins_timeseries.shape[0])).T

    # compute conditional mutual info
    cmi = np.sum(prob[3] * np.log((prob[3] * prob[0]) / (prob[1] * prob[2])))
    
    return cmi

@njit
def II(spins_timeseries, c_matrix):
    """
    Computes interaction information based on mutual information and conditional 
    mutual information for all connected triplets in the network.
    """

    ii_total = 0

    # get all connected pairs
    connected_pairs = np.stack(np.triu(c_matrix).nonzero(), axis=1)

    # loop over pairs
    for pair in connected_pairs:

        x,y = pair

        # compute mutual info for pair
        m_i = mutual_info(pair, spins_timeseries)

        # get all z that form a triplet with pair
        z_x = connected_pairs[connected_pairs[:, 0]==x, 1]
        z_y = connected_pairs[connected_pairs[:, 0]==y, 1]
        z_list = np.intersect1d(z_x, z_y)

        # loop over triplets with these pairs
        for z in z_list:

            # compute conditional mutial info
            cm_i = con_mutual_info(np.array([x,y,z]), spins_timeseries)

            # add interaction information to total
            ii_total += (m_i - cm_i)

    return ii_total

def plot_results(sim_data, T_list, sim_name, save=False, figsize=[6,4]):
    """Plots the results of a full simulation."""
    means_mag, stds_mag, means_sus, stds_sus = sim_data
    lower_bound = np.subtract(means_mag, stds_mag)
    upper_bound = np.add(means_mag, stds_mag)
    plt.figure(figsize=figsize, dpi=300)
    plt.plot(T_list, means_mag)
    plt.fill_between(T_list, lower_bound, upper_bound, alpha=0.3)
    plt.xlabel('T')
    plt.ylabel('M')
    plt.grid()

    if save:
        plt.savefig(f'Results/{sim_name}_M.png', bbox_inches='tight')
    plt.show()    
    
    lower_bound = np.subtract(means_sus, stds_sus)
    upper_bound = np.add(means_sus, stds_sus)
    plt.figure(figsize=figsize, dpi=300)
    plt.plot(T_list, means_sus)
    plt.fill_between(T_list, lower_bound, upper_bound, alpha=0.3)
    plt.xlabel('T')
    plt.ylabel(r'$\chi$')
    plt.grid()
    # idx = np.argmax(sim_data[2])
    # plt.vlines(T_list[idx],  min(sim_data[2]), max(sim_data[2])*1.2, linestyles='dashed', color = 'r')

    if save:
        plt.savefig(f'Results/{sim_name}_sus.png', bbox_inches='tight')
    plt.show()
