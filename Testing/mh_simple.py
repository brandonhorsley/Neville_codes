import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


# basic implementation of Metropolis-Hastings to sample from posterior distribution
# to guess the mean of a simple normal distribution

mu = 0.1  # mean of the distribution that we are trying to guess
sigma = 1  # fixed parameter for the simulation
n = 1  # number of data samples
Y = np.random.normal(loc=mu, scale=sigma, size=n)  # generate the data
# x is parameter we want to estimate, y is data
likelihood = lambda y, x: scipy.stats.norm.pdf(y, loc=x, scale=sigma)  # likelihood formula
prior = lambda x: scipy.stats.uniform.pdf(x)  # prior on mu
posterior = lambda x, y: likelihood(y, x)*prior(x)  # target distribution: posterior on mu given data

# proposal for point x2 from point x1
proposal_distrib = lambda x1, x2: scipy.stats.norm.pdf(x2, loc=x1, scale=sigma)
proposal_draw = lambda x1: np.random.normal(loc=x1, scale=sigma)

N = 10000  # number of iterations

x = 0.0
L = [x]
for _ in range(N):
    u = np.random.uniform()
    x2 = proposal_draw(x1=x)

    L2 = posterior(y=Y[0], x=x2)*proposal_distrib(x2=x, x1=x2)
    L1 = posterior(y=Y[0], x=x)*proposal_distrib(x2=x2, x1=x)

    A = np.min([1, L2/L1])
    if u < A:
        x = x2
    L.append(x)

print(L)
