import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import scipy.stats
import scipy.integrate

# np.random.seed(150)  # same data between two runs

# basic implementation of Metropolis-Hastings to sample from posterior distribution
# to guess the mean of a simple normal distribution

mu = 0.21  # mean of the distribution that we are trying to guess
sigma = 1  # fixed parameter for the simulation
n = 10000  # number of data samples
Y = np.random.normal(loc=mu, scale=sigma, size=(n, ))  # generate the data

N = 5000  # number of iterations for MCMC

# In the functions, x is the parameter we want to estimate, y is data (possibly arrays for vectorisation)

# likelihood formula up to constant prefactor (normalisation)
# Y: data shape (n, )
# x: scalar parameter we want to estimate, shape (m, )
def likelihood(y, x):
    z = np.exp(-(y[:, np.newaxis] - x[np.newaxis, :]) ** 2 / (2 * sigma ** 2))
    z = np.prod(z, axis=0)  # product for iid
    return z


def loglikelihood(y, x):
    z = -(y[:, np.newaxis] - x[np.newaxis, :]) ** 2 / (2 * sigma ** 2)
    z = np.sum(z, axis=0)  # product for iid
    return z


# prior on mu, uniform between -1 and 1
# x: shape (m, )
def prior(x):
    return scipy.stats.uniform.pdf(x, loc=-1, scale=2)


def logprior(x):
    return scipy.stats.uniform.logpdf(x, loc=-1, scale=2)


# target distribution: posterior on mu given data, up to constant prefactor
# x: unknown parameter we want to estimate, shape (m, )
# y: data drawn from likelihood distribution, shape (n, )
def posterior(x, y):
    return likelihood(y, x) * prior(x)


def logposterior(x, y):
    return loglikelihood(y, x) + logprior(x)


# Proposal density to get point x2 from normal distribution centered on point x1
def proposal_distribution(x1, x2):
    return scipy.stats.norm.pdf(x2, loc=x1, scale=sigma)


def logproposal_distribution(x1, x2):
    return scipy.stats.norm.logpdf(x2, loc=x1, scale=sigma)


# draw a single sample from proposal; normal distribution centred on point x1
def proposal_draw(x1, size=(1, )):
    return np.random.normal(loc=x1, scale=sigma, size=size)


x = 0.0  # initial point
L = [x]  # output chain
for _ in range(N):
    u = np.random.uniform()  # uniform draw to test acceptance
    x2 = proposal_draw(x1=x)  # draw a single candidate

    # likelihood ratio
    # L2 = posterior(y=Y, x=x2) * proposal_distribution(x2=x, x1=x2)
    # L1 = posterior(y=Y, x=np.array([x])) * proposal_distribution(x2=x2, x1=np.array([x]))
    # ratio = L2 / L1

    # loglikelihood difference
    LL2 = logposterior(y=Y, x=x2) + logproposal_distribution(x2=x, x1=x2)
    LL1 = logposterior(y=Y, x=np.array([x])) + logproposal_distribution(x2=x2, x1=np.array([x]))
    LLdiff = LL2 - LL1

    # likelihood ratio
    # A = np.where(ratio > 1, 1, ratio)
    # if (u < A)[0]:  # TODO array version, likely need cleanup
    #     x = x2[0]
    # L.append(x)

    # loglikelihood difference
    A = np.where(LLdiff >= 0, 0, LLdiff)
    if (u < np.exp(A))[0]:  # TODO array version, likely need cleanup
        x = x2[0]
    L.append(x)


warmup = N // 10
L = L[warmup:]  # dump first 10% of the chain

# Data visualisation
kernel = scipy.stats.gaussian_kde(L)
mu_range = np.linspace(-2, 2, 400)
estimated_pdf = kernel.evaluate(mu_range)  # already normalised

# plot exact posterior pdf when we use likelihood
# exact_pdf = posterior(mu_range, Y)
# exact_norm = scipy.integrate.quad(lambda x: posterior(np.array([x]), Y)[0], -2, 2)[0]
# exact_pdf /= exact_norm  # normalisation for simple example

exact_logpdf = logposterior(mu_range, Y)
# exact_lognorm = np.log(scipy.integrate.quad(lambda x: posterior(np.array([x]), Y)[0], -2, 2)[0])
# exact_logpdf -= exact_lognorm  # normalisation for simple example
exact_logpdf += np.log(np.max(estimated_pdf)) -np.max(exact_logpdf)  # normalisation to match max of estimated and exact
exact_pdf = np.exp(exact_logpdf)
print(exact_pdf)

print("Data:")
print(Y)
print("Mean:", np.mean(Y))
print()
print("mu for max exact posterior: ", mu_range[np.argmax(exact_pdf)])
print("mu for max estim posterior: ", mu_range[np.argmax(estimated_pdf)])
print("mu exact: ", mu)


plt.figure()
plt.plot(mu_range, estimated_pdf, "-", label="Kernel", color="red")
plt.plot(mu_range, exact_pdf, "-", label="Exact posterior", color="black")
plt.axvline(x=mu, label="Exact mu", color="blue")
plt.legend()
plt.xlim([-1, 1])
plt.ylim([0, np.round(1.2*np.max([exact_pdf, estimated_pdf]))])
plt.show()
