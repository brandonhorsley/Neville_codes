import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

# One more with a single heater and unknown beamsplitter: aV**2 + b, we want to estimate a, b, eta
# we get data by interfering 2 coherent states on an unknown beamsplitter

a = 0.211 * np.pi  # arbitrary value for testing (in rad V^-2)
b = -0.113 * np.pi  # arbitrary value for testing (in rad)
eta = 0.521  # arbitrary value for testing

mu = 0.5  # coherent state intensity
n = 10000  # number of data sample tuples (outcome, voltage)
N = 50000  # size of MCMC

sigma_hyper = 0.5  # variance for proposal distribution

# In the formulas, Y is the data tuple (outcome, voltage), X is the parameters we want to estimate (a, b, eta)

# x: m value array for parameter a, b, eta, shape (m, 3)
# n: number of data sample to draw
# return array of n data points with outcome and voltage (y, v), shape (n, 2)
def data_draw(x, n):
    v = np.random.uniform(size=(n, ))  # draw voltage at random
    v = v[:, np.newaxis].repeat(x.shape[0], axis=1)  # duplicate the value of voltage for each value of a, b, eta
    phi = x[np.newaxis, :, 0] * v ** 2 + x[np.newaxis, :, 1]  # a * V^2 + b, shape (n, m)
    exp_phi = np.exp(1j * phi)
    sqrt_eta = np.sqrt(x[np.newaxis, :, 2])
    sqrt_1eta = np.sqrt(1-x[np.newaxis, :, 2])
    z1 = np.exp(-mu * np.abs(sqrt_eta*exp_phi + sqrt_1eta) ** 2)  # click probability, shape (n, m)
    Z1 = np.concatenate([1 - z1[np.newaxis], z1[np.newaxis]], axis=0)
    z2 = np.exp(-mu * np.abs(sqrt_1eta*exp_phi - sqrt_eta) ** 2)  # click probability
    Z2 = np.concatenate([1 - z2[np.newaxis], z2[np.newaxis]], axis=0)
    Z = Z1[:, np.newaxis] * Z2[np.newaxis, :]  # shape (2,2,n,m)
    Z = Z.reshape((4, Z.shape[2], Z.shape[3]))  # proba distribution
    z = np.cumsum(Z, axis=0)  # cumulative distribution

    u = np.random.uniform(size=(Z.shape[-2], Z.shape[-1]))  # random draw to choose outcome, shape (n,m)

    y = np.argmax(u[np.newaxis, :, :] < z, axis=0)  # selected outcome value, shape (n, m)
    Y = np.concatenate([y[:, :, np.newaxis], v[:, :, np.newaxis]], axis=-1)  # shape (n,m,2)
    return Y


# loglikelihood formula
# Y: data shape (n, 2)
# x: scalar parameter we want to estimate, shape (m, 3)
def loglikelihood(y, x):
    phi = x[np.newaxis, :, 0] * y[:, np.newaxis, 1]**2 + x[np.newaxis, :, 1]  # a * V^2 + b, shape (n, m)
    exp_phi = np.exp(1j*phi)
    sqrt_eta = np.sqrt(np.abs(x[np.newaxis, :, 2]))  # abs to avoid negative number under sqrt for computation
    sqrt_1eta = np.sqrt(np.abs(1 - x[np.newaxis, :, 2]))  # anyways the prior will exclude the point if out of bounds
    z1 = np.exp(-mu * np.abs(sqrt_eta * exp_phi + sqrt_1eta) ** 2)  # click probability, shape (n, m)
    Z1 = np.concatenate([1 - z1[np.newaxis], z1[np.newaxis]], axis=0)
    z2 = np.exp(-mu * np.abs(sqrt_1eta * exp_phi - sqrt_eta) ** 2)  # click probability
    Z2 = np.concatenate([1 - z2[np.newaxis], z2[np.newaxis]], axis=0)
    Z = Z1[:, np.newaxis] * Z2[np.newaxis, :]  # shape (2,2,n,m)
    Z = Z.reshape((4, Z.shape[2], Z.shape[3]))
    mask = np.array(y[:, 0], dtype=int)
    z = Z[mask, np.arange(Z.shape[1])]  # select only the outcomes that we observe
    z += scipy.stats.uniform.logpdf(y[:, np.newaxis, 1])
    z = np.sum(np.log(z), axis=0)
    return z


# prior on a, uniform between 0 and 1
# prior on b, uniform between -1 and 1
# prior on eta, uniform between 0 and 1
# x: shape (m, )
# return: evaluation of logpdf of prior at each point, shape (m, )
def logprior(x):
    x0 = scipy.stats.uniform.logpdf(x[:, 0], loc=0, scale=1)
    x1 = scipy.stats.uniform.logpdf(x[:, 1], loc=-1, scale=2)
    x2 = scipy.stats.uniform.logpdf(x[:, 2], loc=0, scale=1)
    return x0 + x1 + x2


# target distribution: logposterior on a, b, eta given data
# x: unknown parameter we want to estimate, shape (m, index)
# y: data drawn from likelihood distribution, shape (n, )
# return: evaluation of posterior distribution for each parameter value, shape (m1, m2)
def logposterior(x, y):
    return loglikelihood(y, x) + logprior(x)


# Proposal density: a1 to a2  and b1 to b2 normal distribution
# x1: value of parameters of initial point, shape (m, index)
# x2: value of parameters of final point, shape (m, index)
# return: evaluation of the pdf of reaching final point from initial point for each value of parameter, shape (m, )
def logproposal_distribution(x1, x2):
    p_0 = scipy.stats.norm.logpdf(x2[:, 0], loc=x1[:, 0], scale=sigma_hyper)
    p_1 = scipy.stats.norm.logpdf(x2[:, 1], loc=x1[:, 1], scale=sigma_hyper)
    p_2 = scipy.stats.norm.logpdf(x2[:, 2], loc=x1[:, 2], scale=sigma_hyper)
    return p_0 + p_1 + p_2


# Draw a single sample from proposal; normal distribution centred on point x1
# x1: value of parameters, shape (m1, m2, index)
# return: value of parameters for final point x2, same shape as x1
def proposal_draw(x1):
    a_draw = np.random.normal(loc=x1[:, 0], scale=sigma_hyper)
    b_draw = np.random.normal(loc=x1[:, 1], scale=sigma_hyper)
    eta_draw = np.random.normal(loc=x1[:, 2], scale=sigma_hyper)
    return np.concatenate([a_draw[:, np.newaxis], b_draw[:, np.newaxis], eta_draw[:, np.newaxis]], axis=-1)


y = data_draw(x=np.array([[a, b, eta]]), n=n).reshape((-1, 2))  # draw data from exact model
x = np.array([[0.1, 0.1, 0.5]])  # initial point
L = [x]  # output chain
for _ in range(N):
    u = np.random.uniform()  # uniform draw to test acceptance
    x2 = proposal_draw(x1=x)  # draw a single candidate

    LL2 = logposterior(y=y, x=x2) + logproposal_distribution(x2=x, x1=x2)
    LL1 = logposterior(y=y, x=x) + logproposal_distribution(x2=x2, x1=x)
    LLdifference = LL2 - LL1
    A = np.where(LLdifference >= 0, 0, LLdifference)
    if u <= np.exp(A):  # TODO array version, likely need cleanup
        x = x2
    L.append(x)
warmup = N // 10
L = L[warmup:]  # dump first 10% of the chain
chain = np.array(L)
chain = chain.reshape((-1, 3))
chain = chain.transpose()


# Data visualisation

# marginal in a
kernel = scipy.stats.gaussian_kde(chain[0])
a_range = np.linspace(-1, 1, 1000)
estimated_pdf = kernel.evaluate(a_range)  # already normalised

print("a for MAP estimator: ", a_range[np.argmax(estimated_pdf)])
print("a for MMSE estimator: ", np.mean(chain[0]))
print("a exact: ", a)
print()

plt.figure()
plt.plot(a_range, estimated_pdf, "-", label="Kernel", color="red")
plt.axvline(x=a, label="Exact a", color="blue")
plt.legend()
plt.xlim([0, 1])
plt.ylim([0, np.round(1.2*np.max([estimated_pdf]))])
plt.show()


# marginal in b
kernel = scipy.stats.gaussian_kde(chain[1])
b_range = np.linspace(-1, 1, 1000)
estimated_pdf = kernel.evaluate(b_range)  # already normalised

print("b for MAP estimator: ", b_range[np.argmax(estimated_pdf)])
print("b for MMSE estimator: ", np.mean(chain[1]))
print("b exact: ", b)


plt.figure()
plt.plot(b_range, estimated_pdf, "-", label="Kernel", color="red")
plt.axvline(x=b, label="Exact b", color="blue")
plt.legend()
plt.xlim([-1, 1])
plt.ylim([0, np.round(1.2*np.max([estimated_pdf]))])
plt.show()


# marginal in eta
kernel = scipy.stats.gaussian_kde(chain[2])
eta_range = np.linspace(0, 1, 1000)
estimated_pdf = kernel.evaluate(eta_range)  # already normalised

print("eta for MAP estimator: ", eta_range[np.argmax(estimated_pdf)])
print("eta for MMSE estimator: ", np.mean(chain[2]))
print("eta exact: ", eta)


plt.figure()
plt.plot(eta_range, estimated_pdf, "-", label="Kernel", color="red")
plt.axvline(x=eta, label="Exact eta", color="blue")
plt.legend()
plt.xlim([0, 1])
plt.ylim([0, np.round(1.2*np.max([estimated_pdf]))])
plt.show()