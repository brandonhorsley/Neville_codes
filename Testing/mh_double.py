import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import scipy.stats
import scipy.integrate

# np.random.seed(150)  # same data between two runs

# basic implementation of Metropolis-Hastings to sample from posterior distribution
# to guess the mean and variance of a simple normal distribution

sigma_hyper = 0.45  # hyper parameter of normal distribution to propose next state

mu = 0.4123  # mean of the distribution that we are trying to guess
sigma = 0.345  # std deviation of the distribution we are trying to guess

n = 1000  # number of data samples
Y = np.random.normal(loc=mu, scale=sigma, size=(n, ))  # generate the data
N = 5000  # number of iterations

# In the functions, x is the parameter we want to estimate, y is data (possibly arrays for vectorisation)

# loglikelihood formula up (here the normalisation prefactor is important because it depends on sigma
# y: data shape (n, ), corresponding to outcome value
# x: scalar parameter we want to estimate, shape (m1, m2, index), corresponding to mu, sigma, index
# output: evaluation of likelihood at each point, shape (m1, m2)
def loglikelihood(y, x):
    z = -(y[:, np.newaxis, np.newaxis] - x[np.newaxis, :, :, 0]) ** 2 / (2 * x[np.newaxis, :, :, 1] ** 2)
    z += -0.5*np.log(2*np.pi*x[np.newaxis, :, :, 1]**2)
    z = np.sum(z, axis=0)  # product for iid
    return z


# prior on mu, uniform between -1 and 1
# prior on sigma, uniform between 0 and 1
# x: shape (m1, m2)
# return: evaluation of logpdf of prior at each point, shape (m1, m2)
def logprior(x):
    x1 = scipy.stats.uniform.logpdf(x[:, :, 0], loc=-1, scale=2)
    x2 = scipy.stats.uniform.logpdf(x[:, :, 1], loc=0, scale=1)
    return x1 + x2


# target distribution: logposterior on mu, sigma given data, up to constant term
# x: unknown parameter we want to estimate, shape (m1, m2, index)
# y: data drawn from likelihood distribution, shape (n, )
# return: evaluation of posterior distribution for each parameter value, shape (m1, m2)
def logposterior(x, y):
    return loglikelihood(y, x) + logprior(x)


# Proposal density: mu1 to mu2  and sigma1 to sigma2 normal distribution
# x1: value of parameters of initial point, shape (m1, m2, index)
# x2: value of parameters of final point, shape (m1, m2, index)
# return: evaluation of the pdf of reaching final point from initial point for each value of parameter, shape (m1, m2)
def logproposal_distribution(x1, x2):
    p_mu = scipy.stats.norm.logpdf(x2[:, :, 0], loc=x1[:, :, 0], scale=sigma_hyper)
    p_sigma = scipy.stats.norm.logpdf(x2[:, :, 1], loc=x1[:, :, 1], scale=sigma_hyper)
    return p_mu + p_sigma


# Draw a single sample from proposal; normal distribution centred on point x1
# x1: value of parameters, shape (m1, m2, index)
# return: value of parameters for final point x2, same shape as x1
def proposal_draw(x1):
    mu_draw = np.random.normal(loc=x1[:, :, 0], scale=sigma_hyper)
    sigma_draw = np.random.normal(loc=x1[:, :, 1], scale=sigma_hyper)
    return np.concatenate([mu_draw[:, :, np.newaxis], sigma_draw[:, :, np.newaxis]], axis=-1)


x = np.array([[[0.0, 0.5]]])  # initial point
L = [x]  # output chain
for _ in range(N):
    u = np.random.uniform()  # uniform draw to test acceptance
    x2 = proposal_draw(x1=x)  # draw a single candidate

    LL2 = logposterior(y=Y, x=x2) + logproposal_distribution(x2=x, x1=x2)
    LL1 = logposterior(y=Y, x=x) + logproposal_distribution(x2=x2, x1=x)
    LLdifference = LL2 - LL1
    A = np.where(LLdifference >= 0, 0, LLdifference)
    if u <= np.exp(A):  # TODO array version, likely need cleanup
        x = x2
    L.append(x)
warmup = N // 10
L = L[warmup:]  # dump first 10% of the chain
chain = np.array(L)
chain = chain.reshape((-1, 2))
chain = chain.transpose()


# Data visualisation
# joint distribution
# kernel = scipy.stats.gaussian_kde(chain)
# mumin, mumax = -1, 1
# sigmamin, sigmamax = 0.01, 1
# mu_range, sigma_range = np.mgrid[mumin:mumax:100j, sigmamin:sigmamax:100j]
# positions = np.vstack([mu_range.ravel(), sigma_range.ravel()])
# Z = np.reshape(kernel(positions).T, mu_range.shape)
#
# fig, ax = plt.subplots()
# ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[mumin, mumax, sigmamin, sigmamax])
# ax.plot(chain[0], chain[1], 'k.', markersize=2)
# ax.set_xlim([mumin, mumax])
# ax.set_ylim([sigmamin, sigmamax])
# plt.show()


# marginal in mu
kernel = scipy.stats.gaussian_kde(chain[0])
mu_range = np.linspace(-1, 1, 1000)
estimated_pdf = kernel.evaluate(mu_range)  # already normalised

# exact_logpdf = logposterior(mu_range, Y)
# exact_logpdf += np.log(np.max(estimated_pdf)) - np.max(exact_logpdf)  # normalisation to match max of estimated and exact
# exact_pdf = np.exp(exact_logpdf)

print("Data mean:", np.mean(Y))
# print("mu for max exact posterior: ", mu_range[np.argmax(exact_pdf)])
print("mu for max estim posterior: ", mu_range[np.argmax(estimated_pdf)])
print("mu exact: ", mu)
print()

plt.figure()
plt.plot(mu_range, estimated_pdf, "-", label="Kernel", color="red")
# plt.plot(mu_range, exact_pdf, "-", label="Exact posterior", color="black")
plt.axvline(x=mu, label="Exact mu", color="blue")
plt.legend()
plt.xlim([-1, 1])
# plt.ylim([0, np.round(1.2*np.max([exact_pdf, estimated_pdf]))])
plt.ylim([0, np.round(1.2*np.max([estimated_pdf]))])
plt.show()


# marginal in sigma
kernel = scipy.stats.gaussian_kde(chain[1])
sigma_range = np.linspace(0.01, 2, 1000)
estimated_pdf = kernel.evaluate(sigma_range)  # already normalised

print("Data std:", np.std(Y))
print("sigma for max estim posterior: ", sigma_range[np.argmax(estimated_pdf)])
print("sigma exact: ", sigma)


plt.figure()
plt.plot(sigma_range, estimated_pdf, "-", label="Kernel", color="red")
plt.axvline(x=sigma, label="Exact sigma", color="blue")
plt.legend()
plt.xlim([0, 2])
# plt.ylim([0, np.round(1.2*np.max([exact_pdf, estimated_pdf]))])
plt.ylim([0, np.round(1.2*np.max([estimated_pdf]))])
plt.show()
