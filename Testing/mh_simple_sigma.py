import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import scipy.stats
import scipy.integrate

# np.random.seed(150)  # same data between two runs

# basic implementation of Metropolis-Hastings to sample from posterior distribution
# to guess the variance of a simple normal distribution

mu = 0.21  # fixed mean of the distribution for the simulation
sigma = 0.756  # standard deviation we are trying to estimate
sigma_hyper = 0.2  # parameter to propose new point
n = 1000  # number of data samples
N = 10000  # number of iterations for MCMC

Y = np.random.normal(loc=mu, scale=sigma, size=(n, ))  # generate the data

# In the functions, x is the parameter we want to estimate, y is data (possibly arrays for vectorisation)

# likelihood formula up to constant prefactor
# Y: data shape (n, )
# x: scalar parameter we want to estimate, shape (m, )
def likelihood(y, x):
    z = np.exp(-(y[:, np.newaxis] - mu) ** 2 / (2 * x[np.newaxis, :] ** 2))/np.sqrt(2*np.pi*x[np.newaxis, :]**2)
    z = np.prod(z, axis=0)  # product for iid
    return z


def loglikelihood(y, x):
    z = -(y[:, np.newaxis] - mu) ** 2 / (2 * x[np.newaxis, :] ** 2)
    z += -0.5*np.log(2*np.pi*x[np.newaxis, :]**2)
    z = np.sum(z, axis=0)  # product for iid
    return z


# prior on sigma, uniform between 0 and 2
# x: shape (m, )
def prior(x):
    return scipy.stats.uniform.pdf(x, loc=0, scale=2)


def logprior(x):
    return scipy.stats.uniform.logpdf(x, loc=0, scale=2)


# target distribution: posterior on mu given data, up to constant prefactor
# x: unknown parameter we want to estimate, shape (m, )
# y: data drawn from likelihood distribution, shape (n, )
def posterior(x, y):
    return likelihood(y, x) * prior(x)


def logposterior(x, y):
    return loglikelihood(y, x) + logprior(x)


# Proposal density to get point x2 from normal distribution centered on point x1
def proposal_distribution(x1, x2):
    return scipy.stats.norm.pdf(x2, loc=x1, scale=sigma_hyper)


def logproposal_distribution(x1, x2):
    return scipy.stats.norm.logpdf(x2, loc=x1, scale=sigma)


# draw a single sample from proposal; normal distribution centred on point x1
def proposal_draw(x1, size=(1, )):
    return np.random.normal(loc=x1, scale=sigma, size=size)


x = 0.5  # initial point
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
chain = np.array(L)

# Data visualisation
kernel = scipy.stats.gaussian_kde(L)
sigma_range = np.linspace(0.01, 2, 400)
estimated_pdf = kernel.evaluate(sigma_range)  # already normalised

# exact_pdf = posterior(mu_range, Y)
# exact_norm = scipy.integrate.quad(lambda x: posterior(np.array([x]), Y)[0], -2, 2)[0]
# exact_pdf /= exact_norm  # normalisation for simple example

exact_logpdf = logposterior(sigma_range, Y)
# exact_lognorm = np.log(scipy.integrate.quad(lambda x: posterior(np.array([x]), Y)[0], -2, 2)[0])
# exact_logpdf -= exact_lognorm  # normalisation for simple example
exact_logpdf += np.log(np.max(estimated_pdf)) - np.max(exact_logpdf)  # normalisation to match max of estimated and exact
exact_pdf = np.exp(exact_logpdf)

print("Std of data:", np.std(Y))
print("mu for max exact posterior: ", sigma_range[np.argmax(exact_pdf)])
print("mu for max estim posterior: ", sigma_range[np.argmax(estimated_pdf)])
print("sigma exact: ", sigma)


plt.figure()
plt.plot(sigma_range, estimated_pdf, "-", label="Kernel", color="red")
plt.plot(sigma_range, exact_pdf, "-", label="Exact posterior", color="black")
plt.axvline(x=sigma, label="Exact sigma", color="blue")
plt.legend()
plt.xlim([0, 2])
plt.ylim([0, np.round(1.2*np.max([exact_pdf, estimated_pdf]))])
# plt.ylim([0, np.round(1.2*np.max([estimated_pdf]))])
plt.show()
