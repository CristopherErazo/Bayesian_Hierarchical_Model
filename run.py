# Script to run the sampler in the cluster

# Libraries
import numpy as np
import os 
import emcee
import time
import Module as md

path_data = f'Data/'
os.makedirs(path_data,exist_ok=True)

# DEFINE THE HYPERPARAMETERS AND SAMPLE FROM THE BHM

# Model Parameters
J = 10                   # Number of hospitals   
m , s = 3 , 0.2               # Parameters for P(mu_0)
alpha , beta = 3 , 0.5        # Parameters for P(sigma2_0)
frequencies = [12,24,52]    # Possible reporting frequencies
seed = 2001                # Seed for reproducibility

# Get samples from the hierarchical model
ns , mu_0 , sigma2_0 , lambdas , Y = md.sample_hierarchical_model(J,frequencies,m,s,alpha,beta,seed=seed)

print(f'Number of hospitals: {J}')
print(f'Number of samples: {ns}')
print(f'Parameters for mu_0: m={m}, s={s}')
print(f'Parameters for sigma2_0: alpha={alpha}, beta={beta}')
print(f'Frequencies: {frequencies}')
print(f'Seed: {seed}')
print(f'mu_0: {mu_0:.3}')
print(f'sigma2_0: {sigma2_0:.3}')
lam_str = ', '.join([f'{l:.3}' for l in lambdas])
print(f'lambda: {lam_str}')


# SAMPLE FROM THE  POSTERIOR DISTRIBUTION 

# Parameters of the sampler
n_walkers = 40
burnin = 100
nsteps = 500

# Moves allows to define the type of proposal move for the walkers.
moves=[(emcee.moves.DEMove(), 0.8),
       (emcee.moves.DESnookerMove(gammas=0.5), 0.2)]


# Define the log posterior functions available
functions = [md.log_full_posterior,
             md.log_marginal_posterior_no_pooling,
             md.log_marginal_posterior_pooling]

logP = functions[0]
name_fun = logP.__name__

# Define the dimension of the problem and full flag
if 'full' in name_fun:
    full = True
    ndim = J + 2   
else:
    full = False
    ndim = J   


# Sample from the full posterior distribution
time1 = time.time()
samples , acc_fraction , autocorr_time = md.emcee_sampling(n_walkers,ndim,logP,burnin,
                                            nsteps,moves,Y,ns,J,m,s,alpha,beta,show=False,
                                            progress=False,full=full)
time2 = time.time()
dt = time2-time1
print(f'Computation time: {dt:.5} seconds = {dt/60:.5} minutes')

# Save the samples
fname = path_data + f'samples_{name_fun}_J{J}_m{m}_s{s}_alpha{alpha}_beta{beta}_seed{seed}_nw{n_walkers}_nburn{burnin}_nsteps{nsteps}'
np.save(fname,samples)