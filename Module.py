import numpy as np
from scipy.integrate import quad
from scipy.stats import invgamma , norm , poisson
from IPython.display import display, Math
import emcee
from emcee.autocorr import AutocorrError

def f(x, x0, w2, alpha, J):
    '''
    Computes the function inside of the Gaussian integral 
    corresponds to equation (9) in the pdf: 
    f(x,lambda) = 1/(x - x0^2 + w^2)^(alpha + J/2)

    Inputs:
    - x: float, variable of the function
    - x0: float, center of curve
    - w2: float, width of the function
    - alpha: float, hyperparameter of the prior distribution
    - J: int, number of Hospitals

    Outputs:
    - f: float, value of the function at x
    '''
    return 1 / ((x - x0)**2 + w2)**(alpha+0.5*J) 

def integrand(x, x0, w2, alpha, J):
    '''
    Computes the full integrand of equation (9) in the pdf
    including the normal distribution.
    Integrand = Normal(x) * f(x,lambda)

    Inputs:
    - x: float, variable of the function
    - x0: float, center of curve
    - w2: float, width of the function
    - alpha: float, hyperparameter of the prior distribution
    - J: int, number of Hospitals

    Outputs:
    - integrand: float, value of the integrand at x 
    '''
    fun = f(x, x0, w2, alpha, J)
    return norm.pdf(x) * fun

def F(x0, w2, alpha, J):
    '''
    Computes the F(lambda) function which is the Gausian integral
    of equation (9) in the pdf.

    Inputs:
    - x0: float, center of curve
    - w2: float, width of the function
    - alpha: float, hyperparameter of the prior distribution
    - J: int, number of Hospitals

    Outputs:
    - result: float, value of the integral F(lambda)
    - error: float, error of the integral
    '''
    result, error = quad(integrand, -np.inf, np.inf, args=(x0, w2, alpha, J))
    return result , error


def log_likelihood(lambdas , Y , ns , J): 
    '''
    Computes the log likelihood of the vector lambdas given the data
    where the distribution of the data is Poisson
                    yij ~ Poisson(lambda_j)

    Related to equation (12) in the pdf.

    Inputs:
    - lambdas: np.array of floats, lambdas of the hospitals. 
             lambdas = (lambda_j) for j=1,...,J
    - Y: array of np.arrays of ints, number of counts in each hospital. 
             Y = [Yj] for j=1,...,J where
             Yj = [yij] for i=1,...,ns[j]
    - ns: array of ints, number of observations in each hospital. 
             ns[j] = len(Y[j])
    - J: int, number of Hospitals

    Outputs:
    - log_likelihood: float, value of the log likelihood 
    '''
    if (lambdas > 0).all():
        # Compute log_lambdas and the sum of counts in each hospital
        log_lambdas = np.log(lambdas)
        S = np.array([np.sum(Y[j]) for j in range(J)])
        return np.sum( S*log_lambdas - ns*lambdas)
    else:
        return -np.inf

def log_latent_distribution(lambdas , mu_0 , sigma2_0 , J): 
    '''
    Computes the log of the latent distribution of the lambdas
    given the parameters P(lambda|mu_0,sigma2_0) where 
    the distribution of the lambdas is log-normal:
                log lambda_j ~ N(mu_0,sigma2_0)

    Related to equation (12) in the pdf.

    Inputs:
    - lambdas: np.array of floats, lambdas of the hospitals. 
             lambdas = (lambda_j) for j=1,...,J
    - mu_0: float, mean of the prior distribution
    - sigma2_0: float, variance of the prior distribution
    - J: int, number of Hospitals

    Outputs:
    - log_latent_distribution: float, value of the log of the latent distribution
    '''
    if (lambdas > 0).all() and sigma2_0 > 0:
        # Compute log_lambdas
        log_lambdas = np.log(lambdas)
        # Compute the sufficient statistics
        mu_log_lambdas = np.mean(log_lambdas)
        sigma2_log_lambdas = np.var(log_lambdas)
        return -J*( mu_log_lambdas + 0.5*np.log(sigma2_0) +0.5*( (mu_0 - mu_log_lambdas)**2/sigma2_0 ))
    else:
        return -np.inf
    

def log_priors(mu_0 , sigma2_0 , m , s, alpha , beta): 
    '''
    Computes the log-prior of the paramters mu_0 and sigma2_0 where:
                mu_0 ~ N(m,s^2)
        sigma2_0 ~ InverseGamma(alpha,beta)

    Shown in equation 1 of pdf
    Inputs:
    - mu_0: float, mean of the prior distribution
    - sigma2_0: float, variance of the prior distribution
    - m: float, mean of the prior distribution of mu_0
    - s: float, standard deviation of the prior distribution of mu_0
    - alpha: float, shape parameter of the prior distribution of sigma2_0
    - beta: float, scale parameter of the prior distribution of sigma2_0

    Outputs:
    - log_priors: float, value of the log-prior of the parameters
    '''
    if sigma2_0 > 0:
        log_gaussian = -0.5*(mu_0 - m)**2/s**2
        log_invgamma = - (alpha+1)*np.log(sigma2_0) - beta/sigma2_0
        return log_gaussian + log_invgamma
    else: 
        return -np.inf

def log_full_posterior(theta, Y, ns, J, m, s, alpha, beta):
    '''
    Computes the log of the full posterior distribution of the parameters
    theta = (mu_0, sigma2_0, lambdas) given the data Y.

    Inputs:
    - theta: tuple of floats, parameters of the model
    - Y: array of np.arrays of ints, number of counts in each hospital. 
             Y = [Yj] for j=1,...,J where
             Yj = [yij] for i=1,...,ns[j]
    - ns: array of ints, number of observations in each hospital. 
             ns[j] = len(Y[j])
    - J: int, number of Hospitals
    - m: float, mean of the prior distribution of mu_0
    - s: float, standard deviation of the prior distribution of mu_0
    - alpha: float, shape parameter of the prior distribution of sigma2_0
    - beta: float, scale parameter of the prior distribution of sigma2_0

    Outputs:
    - log_full_posterior: float, value of the log of the full posterior distribution
    '''
    # Extract the parameters
    mu_0, sigma2_0 = theta[:2]
    lambdas = theta[-J:]
    # Compute log posterior
    return log_likelihood(lambdas, Y, ns, J) + log_latent_distribution(lambdas, mu_0, sigma2_0, J) + log_priors(mu_0, sigma2_0, m, s, alpha, beta)

def log_marginal_posterior_pooling(lambdas, Y, ns, J, m, s, alpha, beta):
    '''
    Computes the log of the marginal posterior distribution 
    of the variables lambdas given the data Y with pooling.
    P(lambda|Y) = P(Y|lambda)P(lambda)
   
    Inputs:
    - lambdas: tuple of floats, variables of the model
    - Y: array of np.arrays of ints, number of counts in each hospital. 
             Y = [Yj] for j=1,...,J where
             Yj = [yij] for i=1,...,ns[j]
    - ns: array of ints, number of observations in each hospital. 
             ns[j] = len(Y[j])
    - J: int, number of Hospitals
    - m: float, mean of the prior distribution of mu_0
    - s: float, standard deviation of the prior distribution of mu_0
    - alpha: float, shape parameter of the prior distribution of sigma2_0
    - beta: float, scale parameter of the prior distribution of sigma2_0

    Outputs:
    - log_posterior: float, value of the log of the posterior distribution
    '''
    if (lambdas > 0).all():
        # Compute the log likelihood
        log_lik = log_likelihood(lambdas, Y, ns, J)

        # Compute the sufficient statistics
        log_lambdas = np.log(lambdas)
        mu_log_lambdas = np.mean(log_lambdas)
        sigma2_log_lambdas = np.var(log_lambdas)

        # Compute the parameters for the Gaussian integral
        x0 = (mu_log_lambdas - m)/s
        w2 = 2*(beta+0.5*J*(sigma2_log_lambdas))/(J*s**2)
        logF = np.log(F(x0, w2, alpha, J)[0])
        return log_lik -J*mu_log_lambdas + logF
    else:
        return -np.inf
    

def log_marginal_posterior_no_pooling(lambdas, Y, ns, J, m, s, alpha, beta):
    '''
    Computes the log of the marginal posterior distribution 
    of the variables lambdas given the data Y without pooling.
    P_tilda(lambda|Y) = P(Y|lambda)P_tilda(lambda)
   
    Inputs:
    - lambdas: tuple of floats, variables of the model
    - Y: array of np.arrays of ints, number of counts in each hospital. 
             Y = [Yj] for j=1,...,J where
             Yj = [yij] for i=1,...,ns[j]
    - ns: array of ints, number of observations in each hospital. 
             ns[j] = len(Y[j])
    - J: int, number of Hospitals
    - m: float, mean of the prior distribution of mu_0
    - s: float, standard deviation of the prior distribution of mu_0
    - alpha: float, shape parameter of the prior distribution of sigma2_0
    - beta: float, scale parameter of the prior distribution of sigma2_0

    Outputs:
    - log_posterior: float, value of the log of the posterior distribution
    '''
    if (lambdas > 0).all():
        # Compute the log likelihood
        log_lik = log_likelihood(lambdas, Y, ns, J)

        # Compute the sufficient statistics
        log_lambdas = np.log(lambdas)
        mu_log_lambdas = np.mean(log_lambdas)
        sigma2_log_lambdas = np.var(log_lambdas)

        # Compute the several Gaussian integrals
        w2 = 2*beta/(J*s**2)
        logF_tilde = 0
        for j in range(J):
            x0 = (log_lambdas[j] - m)/s
            logF_tilde += np.log(F(x0, w2, alpha, 1)[0])

        return log_lik -J*mu_log_lambdas + logF_tilde
    else:
        return -np.inf




def sample_hierarchical_model(J, frequencies , m , s , alpha , beta , seed=12345): 
    '''
    Create a sample from the hierarchical model with the following structure:
    1. Sample Reporting Frequencies: ns[j] ~ U{frequencies} for j=1,...,J
    2. Sample Population Level Parameters: mu_0 ~ N(m,s^2)  and sigma_0 ~ InvGamma(alpha,beta)
    3. Sample Hospital Level Parameters: log lambda_j ~ N(mu_0,sigma_0) for j=1,...,J 
    4. Sample Data: y_ij ~ Poisson(lambda_j) for i=1,...,ns[j] and j=1,...,J

    Inputs:
    - J: int, number of Hospitals    
    - frequencies: list of ints, possible reporting frequencies  
    - m: float, mean of the prior distribution of mu_0  
    - s: float, standard deviation of the prior distribution of mu_0    
    - alpha: float, shape parameter of the prior distribution of sigma2_0  
    - beta: float, scale parameter of the prior distribution of sigma2_0  
    - seed: int, seed for replicability  

    Outputs:  
    - ns: array of ints, number of observations in each hospital. 
    - mu_0: float, mean of the prior distribution  
    - sigma2_0: float, variance of the prior distribution 
    - lambdas: np.array of floats, lambdas of the hospitals.   
             lambdas = (lambda_j) for j=1,...,J 
    - Y: array of np.arrays of ints, number of counts in each hospital.  
                Y = [Yj] for j=1,...,J where  
                Yj = [yij] for i=1,...,ns[j]  
    '''
    
    # Fix seed for replicability
    np.random.seed(seed)
    
    # Sample Reporting Frequencies
    ns = np.random.choice(frequencies,J)
    idx = np.argsort(ns)

    # Sample Hyperparameters
    s2 = s**2
    mu_0 = norm.rvs(loc=m,scale=np.sqrt(s2))
    sigma2_0 = invgamma.rvs(a=alpha,scale=beta)

    # Sample Hospital Level Parameters
    log_lambdas = norm.rvs(loc=mu_0,scale=np.sqrt(sigma2_0),size=J)
    lambdas = np.exp(log_lambdas)

    # Order the variables once sampled just for organization
    ns = ns[idx]
    lambdas = lambdas[idx]

    # Sample data
    Y = [poisson.rvs(mu=r,size= ns[j]) for j,r in enumerate(lambdas)]

    return ns, mu_0, sigma2_0, lambdas, Y


def display_configuration(ns, mu_0, sigma2_0, lambdas , J,alpha,beta,m,s):
    '''
    Print the sampled configuration for the BHM
    '''
    print(f'Number of Hospitals {J = }')

    print('\n Prior Hyperparameters:')
    display(Math(r'\text Normal: \;\; m = {:.2f}, \; s = {:.2f}'.format(m, s)))
    display(Math(r'\text Inv-Gamma: \;\; \alpha = {:.2f}, \; \beta = {:.2f}'.format(alpha, beta)))

    print(f'\nReporting Frequencies:')
    display(Math(r'(n_j)_{j\leq J} = (' + ', \\;'.join(['{:}'.format(nj) for nj in ns]) + ')'))

    
    print('\nPopulation Level Parameters:')
    display(Math(r'\mu_0 = {:.2f}, \; \sigma_0 = {:.3f}'.format(mu_0, sigma2_0)))

    print('\nHospital Level Parameters:')
    display(Math(r'(\lambda_j)_{j\leq J} = (' + ', \\;'.join(['{:.2f}'.format(r) for r in lambdas]) + ')'))

def get_sample_prior_model(m,s,alpha,beta,J,n_walkers):
    '''
    Sample from the prior distribution of the parameters (mu_0,sigma2_0,lambdas)
    following the BHM to start the Monte Carlo algorithm. 
            mu_0 ~ N(m,s^2)
            sigma2_0 ~ InvGamma(alpha,beta)
            lambda_j ~ N(mu_0,sqrt(sigma2_0)) for j=1,...,J
    Inputs:
    - m: float, mean of the prior distribution of mu_0
    - s: float, standard deviation of the prior distribution of mu_0
    - alpha: float, shape parameter of the prior distribution of sigma2_0
    - beta: float, scale parameter of the prior distribution of sigma2_0
    - J: int, number of Hospitals
    - n_walkers: int, number of walkers in the Monte Carlo algorithm

    Outputs:
    - theta_0 = (mu_0,sigma2_0,lambdas): np.array of floats, shape = (n_walkers,J+2)
    '''
    # Sample from the prior distribution
    mu_0 = norm.rvs(loc=m,scale=s,size=n_walkers)
    sigma2_0 = invgamma.rvs(a=alpha,scale=beta,size=n_walkers)
    # Sample from the hospital level parameters
    lambdas = np.array( [np.exp(norm.rvs(loc=mu_0[k],scale=np.sqrt(sigma2_0[k]), size=J)) for k in range(n_walkers)])
    # Group the samples in a single array
    theta_0 = np.column_stack((mu_0,sigma2_0,lambdas))
    return theta_0



def get_sample_uniform(J,n_walkers,a=1):
    '''
    Sample from a uniform distribution the parameters (mu_0,sigma2_0,lambdas)
    to start the Monte Carlo algorithm. 
            mu_0 ~ U(-a,a)
            sigma2_0 ~ U(0,2a)
            lambda_j ~ U(0,2a) for j=1,...,J
    Inputs:
    - J: int, number of Hospitals
    - n_walkers: int, number of walkers in the Monte Carlo algorithm
    - a: float, range of the uniform distribution

    Outputs:
    - theta_0 = (mu_0,sigma2_0,lambdas): np.array of floats, shape = (n_walkers,J+2)
    '''
    # Sample the population level parameters
    mu_0 = np.random.uniform(-a,a,size=n_walkers)
    sigma2_0 = np.random.uniform(0,2*a,size=n_walkers)
    # Sample from the hospital level parameters
    lambdas = np.random.uniform(0,2*a,size=(n_walkers,J))
    # Group the samples in a single array
    theta_0 = np.column_stack((mu_0,sigma2_0,lambdas))
    return theta_0


def emcee_sampling(n_walkers,ndim,logP,burnin,nsteps,moves,Y,ns,J,m,s,alpha,beta,show=True,progress=True,full=True):
        '''
        Run the emcee sampler to obtain samples from the posterior distribution
        of the parameters given the data Y.

        Inputs:
        - n_walkers: int, number of walkers in the Monte Carlo algorithm
        - ndim: int, number of dimensions of the parameter space
        - logP: function, log-posterior distribution of the parameters
                logP (theta,*args) with args = [Y,ns,J,m,s,alpha,beta]
        - burnin: int, number of steps for the burn-in period
        - nsteps: int, number of steps for the production period
        - moves: list of functions, proposal moves for the walkers with probabilities
        - Y: array of np.arrays of ints, number of counts in each hospital. 
             Y = [Yj] for j=1,...,J where
             Yj = [yij] for i=1,...,ns[j]
        - ns: array of ints, number of observations in each hospital.
                ns[j] = len(Y[j])
        - J: int, number of Hospitals
        - m: float, mean of the prior distribution of mu_0
        - s: float, standard deviation of the prior distribution of mu_0
        - alpha: float, shape parameter of the prior distribution of sigma2_0
        - beta: float, scale parameter of the prior distribution of sigma2_0
        - show: bool, flag to print information about the sampler

        Outputs:
        - samples: np.array of floats, samples from the posterior distribution
        - acc_fraction: float, acceptance ratio of the sampler
        - autocorr_time: float, autocorrelation time of the
        '''

        args = [Y,ns,J,m,s,alpha,beta]
        # Initial point for the sampler obtained from prior in BHM
        if full:
            theta_0 = get_sample_prior_model(m,s,alpha,beta,J,n_walkers)
        else:
            theta_0 = get_sample_prior_model(m,s,alpha,beta,J,n_walkers)[:,-J:]

        # Initialize the Sampler
        sampler = emcee.EnsembleSampler(n_walkers, ndim, 
                                logP, 
                                args=args,
                                moves=moves)
        # moves allows to define the type of proposal move for the walkers 
        # and can help improve the convergence and efficiency of the sampler. 

        # Run the sampler for the burn-in period and restart it
        state = sampler.run_mcmc(theta_0, burnin, progress=progress)
        sampler.reset()

        # Run the sampler for the production period
        state = sampler.run_mcmc(state, nsteps, progress=progress)

        # Collect the samples from the chain
        samples = sampler.get_chain(flat=False)

        # Retrieve information about the sampler
        acc_fraction = np.mean(sampler.acceptance_fraction)

        print('')
        # Check if the autocorrelation time returns a valid value
        try:
            autocorr_time = np.mean(sampler.get_autocorr_time())
        except AutocorrError as err:
            print(f"AutocorrError: {err}")
            autocorr_time = -1.0
            print(f'The autocorrelation time will be set to {autocorr_time}')

        if show:
            print(f'Acceptance Ratio = {acc_fraction:.3}')
            print(f'Autocorrelation Time = {autocorr_time:.3}')
            print(f'\nShape of the full chain {samples.shape = }')

        return samples,acc_fraction,autocorr_time