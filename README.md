# CSS2023 Group 12

Welcome to our git repository with all code to create the plots in our presentation about the application of the Ising model on brain regions and some more side quests ;)

## Getting started

The main code for our project is written in `mcmc.py` and code for running the main experiments can be found in `random_networks.ipynb`. Install the required packages with
```
pip install -r requirements.txt
```
and run the code in the `random_networks.ipynb` Jupyter notebook to recreate our main results. `basic_5_nodes.ipynb` contains the basic usage of our Metropolis algorithm.

## Detailed description

This repository consists of the following folders:
* `Data`
    All required input data.
* `Results`
    All plots and animations that were generated using the code in this repository.
* `Correlations`
    This is our little sidequest on fMRI data. We measured correlations between spins using the Metropolis algorithm and compared the distribution of those correlations to real fMRI data. We then implemented a Hill climbing algorithm to find which connectivity matrix J would produce a correlation distribution that was closest to the fMRI data distribution. The algorithm ended up with an almost fully connected network and didn't really get close to the real data distribution so we focussed on other quests.

And the following files:
* `mcmc.py`
    This file contains all main code for the Ising model, different versions of connectivity matrices, the metropolis algorithm, running simulations, measuring transfer entropy and interaction information. All code is optimized to use Numba and its nopython mode to speed up computations.
* `basic_5_nodes.ipynb`
    The very basic usage of our Metropolis algorithm on the Ising model with a network of 5 nodes to  recreate figures from Popiel et al. (2020)
* `random_networks.ipynb`
    All code to recreate our main results. Code is rather fast for small networks (n=10) and can be used for different sized network by changing n.
* `DTI.ipynb`
    Analysis of the DTI data. We show the connectivity matrix, fit a powerlaw and show a phase transition on the data.
* `animation.ipynb`
    Using NILearn for plotting and animating the Ising model with DTI data using actual coordinates of the brain regions.
* `interaction_info_sidequest.ipynb`
    As transfer entropy only measures information flow, we wanted to implement another measure called interaction information. We did end up implementing it in `mcmc.py` but did not have enough time to fully analyse our model with this measure anymore so we focused on transfer entropy and this became yet another sidequest.


And a little preview of our animations ;)
![image](Results/animation_T_more_than_tc.gif)

#### References
Popiel, Khajehabdollahi, S., Abeyasinghe, P. M., Riganello, F., Nichols, E. S., Owen, A. M., & Soddu, A. (2020). The Emergence of Integrated Information, Complexity, and “Consciousness” at Criticality. Entropy (Basel, Switzerland), 22(3), 339–. https://doi.org/10.3390/e22030339
