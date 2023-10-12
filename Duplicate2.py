"""
Since Duplicate.py seems to have been fleshed out well i am opening this code file as pseudo version control 
to put iteration number in algorithm functions and finally implement the neville protocol (and add/alter 
visualisation to include markov state number plot so we have a full trace plot system in place).

Active notes:

Acknowledged that i have to put in a ReturnLast argument in MCMC functions since most of them don't need to track all the states, just return the last one. So I will do this for all the necessary Neville algorithms but leave the full model ones as they are.

Implemented Neville protocol with good success on a short run, before doing a longer run i would like to implement the Markov state plot

Added markov state plot. All that is left is to add axes labels really. Additionally perhaps input random selection of true values for parameters for more objective evaluation of performance.

Changed phi to a+bV**2 like in Neville thesis. Changed in this and Duplicate.py. Changed back to original since somehow it is leading to worse estimates, could be case-specific so need to investigate further. Worth noting that this quick investigation was with short runs so could be the lack of ample mixing and convergence that could contribute.
"""

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

# One more example with a single heater and unknown beamsplitter: aV**2 + b, we want to estimate a, b, eta
# we get data by interfering 2 coherent states on an unknown beamsplitter

a = 0.211 * np.pi  # arbitrary value for testing (in rad V^-2)
b = -0.113 * np.pi  # arbitrary value for testing (in rad)
eta = 0.521  # arbitrary value for testing

mu = 0.5  # coherent state intensity
n = 10000  # number of data sample tuples (outcome, voltage)
#N = 50000  # size of MCMC
#N=1000
I=[2,500,50,50,500,100,100,100000] #Determines iteration number for each algorithm call
#I=[2,500,50,50,500,100,100,1000] #Smaller MCMC chain for troubleshooting

sigma_hyper = 0.5  # variance for proposal distribution
eta_sigma=0.005
a_sigma=np.pi/200
b_sigma=0.07
# In the formulas, Y is the data tuple (outcome, voltage), X is the parameters we want to estimate (a, b, eta)

# x: m value array for parameter a, b, eta, shape (m, 3)
# n: number of data sample to draw
# return array of n data points with outcome and voltage (y, v) for each value of parameters, shape (n, m, 2)
def data_draw(x, n):
    v = np.random.uniform(size=(n, ))  # draw voltage at random
    v = v[:, np.newaxis].repeat(x.shape[0], axis=1)  # duplicate the value of voltage for each value of a, b, eta
    #phi = x[np.newaxis, :, 0] * v ** 2 + x[np.newaxis, :, 1]  # a * V^2 + b, shape (n, m)
    phi = x[np.newaxis, :, 0] + x[np.newaxis, :, 1] * v ** 2  # a + b*V^2 + b, shape (n, m)
    exp_phi = np.exp(1j * phi)
    # we shouldn't have negative value under sqrt here because data is generated from realistic value
    sqrt_eta = np.sqrt(x[np.newaxis, :, 2])
    sqrt_1eta = np.sqrt(1-x[np.newaxis, :, 2])
    z1 = np.exp(-mu * np.abs(sqrt_eta*exp_phi + sqrt_1eta) ** 2)  # click probability, shape (n, m)
    Z1 = np.concatenate([1 - z1[np.newaxis], z1[np.newaxis]], axis=0)
    z2 = np.exp(-mu * np.abs(sqrt_1eta*exp_phi - sqrt_eta) ** 2)  # click probability 
    Z2 = np.concatenate([1 - z2[np.newaxis], z2[np.newaxis]], axis=0)
    Z = Z1[:, np.newaxis] * Z2[np.newaxis, :]  # shape (2,2,n,m)
    #print(np.shape(Z))
    Z = Z.reshape((4, Z.shape[2], Z.shape[3]))  # proba distribution
    #print(np.shape(Z))
    #print(Z)
    z = np.cumsum(Z, axis=0)  # cumulative distribution
    #print(z)

    u = np.random.uniform(size=(Z.shape[-2], Z.shape[-1]))  # random draw to choose outcome, shape (n,m)

    y = np.argmax(u[np.newaxis, :, :] < z, axis=0)  # selected outcome value, shape (n, m)
    #print(y)
    Y = np.concatenate([y[:, :, np.newaxis], v[:, :, np.newaxis]], axis=-1)  # shape (n,m,2)
    return Y


# loglikelihood formula
# Y: data shape (n, 2)
# x: scalar parameter we want to estimate, shape (m, 3)
def loglikelihood(y, x):
    #phi = x[np.newaxis, :, 0] * y[:, np.newaxis, 1]**2 + x[np.newaxis, :, 1]  # a * V^2 + b, shape (n, m)
    phi = x[np.newaxis, :, 0] + x[np.newaxis, :, 1] * y[:, np.newaxis, 1] ** 2  # a + b*V^2 + b, shape (n, m)
    exp_phi = np.exp(1j*phi)
    # here we can possibly test unrealistic values leading to negative sqrt, so numerical result is unreliable
    # however for such values, the prior should be such that these points will always be rejected in the acceptance test
    sqrt_eta = np.sqrt(np.abs(x[np.newaxis, :, 2]))  # abs to avoid negative number under sqrt for computation
    sqrt_1eta = np.sqrt(np.abs(1 - x[np.newaxis, :, 2]))
    z1 = np.exp(-mu * np.abs(sqrt_eta * exp_phi + sqrt_1eta) ** 2)  # click probability, shape (n, m)
    Z1 = np.concatenate([1 - z1[np.newaxis], z1[np.newaxis]], axis=0)
    z2 = np.exp(-mu * np.abs(sqrt_1eta * exp_phi - sqrt_eta) ** 2)  # click probability
    Z2 = np.concatenate([1 - z2[np.newaxis], z2[np.newaxis]], axis=0)
    Z = Z1[:, np.newaxis] * Z2[np.newaxis, :]  # combine all detection patterns, shape (2,2,n,m)
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
    #x0 = scipy.stats.uniform.logpdf(x[:, 0], loc=0, scale=1)
    x0 = scipy.stats.uniform.logpdf(x[:, 1], loc=-np.pi, scale=2*np.pi)
    #x1 = scipy.stats.uniform.logpdf(x[:, 1], loc=-1, scale=2)
    x1=scipy.stats.norm.logpdf(x[:,0],loc=b,scale=0.07)
    #x2 = scipy.stats.uniform.logpdf(x[:, 2], loc=0, scale=1)
    x2=scipy.stats.norm.logpdf(x[:,0],loc=0.5,scale=0.005)
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
    #p_0 = scipy.stats.norm.logpdf(x2[:, 0], loc=x1[:, 0], scale=sigma_hyper)
    p_0 = scipy.stats.norm.logpdf(x2[:, 0], loc=x1[:, 0], scale=a_sigma)
    #p_1 = scipy.stats.norm.logpdf(x2[:, 1], loc=x1[:, 1], scale=sigma_hyper)
    p_1 = scipy.stats.norm.logpdf(x2[:, 1], loc=x1[:, 1], scale=b_sigma)
    #p_2 = scipy.stats.norm.logpdf(x2[:, 2], loc=x1[:, 2], scale=sigma_hyper)
    p_2 = scipy.stats.norm.logpdf(x2[:, 2], loc=x1[:, 2], scale=eta_sigma)
    return p_0 + p_1 + p_2

def MIS_logproposal_distribution(x1, x2):
    #p_0 = scipy.stats.norm.logpdf(x2[:, 0], loc=x1[:, 0], scale=sigma_hyper)
    p_0 = scipy.stats.uniform.logpdf(x2[:, 0], loc=-np.pi, scale=2*np.pi)
    #p_1 = scipy.stats.norm.logpdf(x2[:, 1], loc=x1[:, 1], scale=sigma_hyper)
    p_1 = scipy.stats.uniform.logpdf(x2[:, 1], loc=-np.pi, scale=2*np.pi)
    #p_2 = scipy.stats.norm.logpdf(x2[:, 2], loc=x1[:, 2], scale=sigma_hyper)
    p_2 = scipy.stats.uniform.logpdf(x2[:, 2], loc=0, scale=1)
    return p_0 + p_1 + p_2

def a_unif_logproposal_distribution(x1,x2):
    #p_0 = scipy.stats.norm.logpdf(x2[:, 0], loc=x1[:, 0], scale=sigma_hyper)
    p_0 = scipy.stats.uniform.logpdf(x2[:, 0], loc=-np.pi, scale=2*np.pi)
    #p_1 = scipy.stats.norm.logpdf(x2[:, 1], loc=x1[:, 1], scale=sigma_hyper)
    p_1 = scipy.stats.norm.logpdf(x2[:, 1], loc=x1[:, 1], scale=b_sigma)
    #p_2 = scipy.stats.norm.logpdf(x2[:, 2], loc=x1[:, 2], scale=sigma_hyper)
    p_2 = scipy.stats.norm.logpdf(x2[:, 2], loc=x1[:, 2], scale=eta_sigma)
    return p_0 + p_1 + p_2

# Draw a single sample from proposal; normal distribution centred on point x1
# x1: value of parameters, shape (m, index)
# return: value of parameters for final point x2, same shape as x1
def proposal_draw(x1):
    #a_draw = np.random.normal(loc=x1[:, 0], scale=sigma_hyper)
    a_draw = np.random.normal(loc=x1[:, 0], scale=a_sigma)
    #b_draw = np.random.normal(loc=x1[:, 1], scale=sigma_hyper)
    b_draw = np.random.normal(loc=x1[:, 1], scale=b_sigma)
    #eta_draw = np.random.normal(loc=x1[:, 2], scale=sigma_hyper)
    eta_draw = np.random.normal(loc=x1[:, 2], scale=eta_sigma)
    return np.concatenate([a_draw[:, np.newaxis], b_draw[:, np.newaxis], eta_draw[:, np.newaxis]], axis=-1)

def MIS_proposal_draw(x1):
    a_draw = np.random.uniform(low=-np.pi, high=np.pi, size=(1,))
    b_draw = np.random.uniform(low=-np.pi, high=np.pi, size=(1,)) #Since there aren't really bounds on 'b' i have given same bounds as 'a'
    eta_draw = np.random.uniform(low=0, high=1, size=(1,))
    return np.concatenate([a_draw[:, np.newaxis], b_draw[:, np.newaxis], eta_draw[:, np.newaxis]], axis=-1)

def a_proposal_draw(x1):
    a_draw = np.random.normal(loc=x1[:, 0], scale=a_sigma)
    b_draw=x1[:,1]
    eta_draw=x1[:,2]
    return np.concatenate([a_draw[:, np.newaxis], b_draw[:, np.newaxis], eta_draw[:, np.newaxis]], axis=-1)

def MIS_a_proposal_draw(x1):
    a_draw = np.random.uniform(low=-np.pi, high=np.pi, size=(1,))
    b_draw=x1[:,1]
    eta_draw=x1[:,2]
    return np.concatenate([a_draw[:, np.newaxis], b_draw[:, np.newaxis], eta_draw[:, np.newaxis]], axis=-1)

def a_unif_proposal_draw(x1):
    a_draw = np.random.uniform(low=-np.pi, high=np.pi, size=(1,))
    b_draw=x1[:,1]
    eta_draw=x1[:,2]
    return np.concatenate([a_draw[:, np.newaxis], b_draw[:, np.newaxis], eta_draw[:, np.newaxis]], axis=-1)


def b_proposal_draw(x1):
    a_draw = x1[:,0]
    b_draw = np.random.normal(loc=x1[:, 1], scale=b_sigma)
    eta_draw = x1[:,2]
    return np.concatenate([a_draw[:, np.newaxis], b_draw[:, np.newaxis], eta_draw[:, np.newaxis]], axis=-1)

def MIS_b_proposal_draw(x1):
    a_draw = x1[:,0]
    b_draw = np.random.uniform(low=-np.pi, high=np.pi, size=(1,))
    eta_draw = x1[:,2]
    return np.concatenate([a_draw[:, np.newaxis], b_draw[:, np.newaxis], eta_draw[:, np.newaxis]], axis=-1)

def eta_proposal_draw(x1):
    a_draw = x1[:, 0]
    b_draw = x1[:,1]
    eta_draw = np.random.normal(loc=x1[:, 2], scale=eta_sigma)
    return np.concatenate([a_draw[:, np.newaxis], b_draw[:, np.newaxis], eta_draw[:, np.newaxis]], axis=-1)

def MIS_eta_proposal_draw(x1):
    a_draw = x1[:, 0]
    b_draw = x1[:,1]
    eta_draw = np.random.uniform(low=0, high=1, size=(1,))
    return np.concatenate([a_draw[:, np.newaxis], b_draw[:, np.newaxis], eta_draw[:, np.newaxis]], axis=-1)

def pi_draw(x1):
    a_draw=np.pi*(np.random.choice(3,size=(1,))-1)
    b_draw = x1[:,1]
    eta_draw = x1[:,2]
    return np.concatenate([a_draw[:, np.newaxis], b_draw[:, np.newaxis], eta_draw[:, np.newaxis]], axis=-1)


y = data_draw(x=np.array([[a, b, eta]]), n=n).reshape((-1, 2))  # draw data from exact model
y = y.reshape((-1, 2))  # reshape to remove the axis for x since we have m=1; not sure if it's relevant to have any m>1
x = np.array([[0.5, 0.1, 0.5]])  # initial point
#x=np.array([[np.pi,0.7,0.5]]) #initial point array to test pi kick alg

#Alg4
def MCMC_MHinGibbs(x,N, ReturnLast=True):
    #MH-within-Gibbs alg
    L = [x]  # output chain
    for _ in range(N):
        for i in range(len(x[0])):
            if i in [0]:
                u = np.random.uniform()  # uniform draw to test acceptance
                x2 = a_proposal_draw(x1=x)  # draw a single candidate

                LL2 = logposterior(y=y, x=x2) + logproposal_distribution(x2=x, x1=x2)
                LL1 = logposterior(y=y, x=x) + logproposal_distribution(x2=x2, x1=x)
                LLdifference = LL2 - LL1
                A = np.where(LLdifference >= 0, 0, LLdifference)
                if u <= np.exp(A):  # TODO array version, likely need cleanup
                    x = x2
            if i in [1]:
                u = np.random.uniform()  # uniform draw to test acceptance
                x2 = b_proposal_draw(x1=x)  # draw a single candidate

                LL2 = logposterior(y=y, x=x2) + logproposal_distribution(x2=x, x1=x2)
                LL1 = logposterior(y=y, x=x) + logproposal_distribution(x2=x2, x1=x)
                LLdifference = LL2 - LL1
                A = np.where(LLdifference >= 0, 0, LLdifference)
                if u <= np.exp(A):  # TODO array version, likely need cleanup
                    x = x2
            if i in [2]:
                u = np.random.uniform()  # uniform draw to test acceptance
                x2 = eta_proposal_draw(x1=x)  # draw a single candidate

                LL2 = logposterior(y=y, x=x2) + logproposal_distribution(x2=x, x1=x2)
                LL1 = logposterior(y=y, x=x) + logproposal_distribution(x2=x2, x1=x)
                LLdifference = LL2 - LL1
                A = np.where(LLdifference >= 0, 0, LLdifference)
                if u <= np.exp(A):  # TODO array version, likely need cleanup
                    x = x2
        L.append(x)
    #Hashing out warmup since need to only take last result for most of these algorithms
    #warmup = N // 10
    #L = L[warmup:]  # dump first 10% of the chain
    if ReturnLast:
        state=L[-1]
        return state
    else:
        chain = np.array(L)
        chain = chain.reshape((-1, 3))
        chain = chain.transpose()  # not very meaningful to transpose, just simpler for kde call later
        return chain

#Alg4_alpha
def MCMC_MHinGibbs_a(x,N, ReturnLast=True):
    #MH-within-Gibbs alg
    L = [x]  # output chain
    for _ in range(N):
        for i in range(len(x[0])):
            if i in [0]:
                u = np.random.uniform()  # uniform draw to test acceptance
                x2 = a_proposal_draw(x1=x)  # draw a single candidate

                LL2 = logposterior(y=y, x=x2) + logproposal_distribution(x2=x, x1=x2)
                LL1 = logposterior(y=y, x=x) + logproposal_distribution(x2=x2, x1=x)
                LLdifference = LL2 - LL1
                A = np.where(LLdifference >= 0, 0, LLdifference)
                if u <= np.exp(A):  # TODO array version, likely need cleanup
                    x = x2
        L.append(x)
    #Hashing out warmup since need to only take last result for most of these algorithms
    #warmup = N // 10
    #L = L[warmup:]  # dump first 10% of the chain
    if ReturnLast:
        state=L[-1]
        return state
    else:
        chain = np.array(L)
        chain = chain.reshape((-1, 3))
        chain = chain.transpose()  # not very meaningful to transpose, just simpler for kde call later
        return chain

#chain=MCMC_MHinGibbs_a(x)

#Alg4_beta
def MCMC_MHinGibbs_b(x,N, ReturnLast=True):
    #MH-within-Gibbs alg
    L = [x]  # output chain
    for _ in range(N):
        for i in range(len(x[0])):
            if i in [0]:
                u = np.random.uniform()  # uniform draw to test acceptance
                x2 = a_proposal_draw(x1=x)  # draw a single candidate

                LL2 = logposterior(y=y, x=x2) + logproposal_distribution(x2=x, x1=x2)
                LL1 = logposterior(y=y, x=x) + logproposal_distribution(x2=x2, x1=x)
                LLdifference = LL2 - LL1
                A = np.where(LLdifference >= 0, 0, LLdifference)
                if u <= np.exp(A):  # TODO array version, likely need cleanup
                    x = x2
            if i in [1]:
                u = np.random.uniform()  # uniform draw to test acceptance
                x2 = b_proposal_draw(x1=x)  # draw a single candidate

                LL2 = logposterior(y=y, x=x2) + logproposal_distribution(x2=x, x1=x2)
                LL1 = logposterior(y=y, x=x) + logproposal_distribution(x2=x2, x1=x)
                LLdifference = LL2 - LL1
                A = np.where(LLdifference >= 0, 0, LLdifference)
                if u <= np.exp(A):  # TODO array version, likely need cleanup
                    x = x2
        L.append(x)
    #Hashing out warmup since need to only take last result for most of these algorithms
    #warmup = N // 10
    #L = L[warmup:]  # dump first 10% of the chain
    if ReturnLast:
        state=L[-1]
        return state
    else:
        chain = np.array(L)
        chain = chain.reshape((-1, 3))
        chain = chain.transpose()  # not very meaningful to transpose, just simpler for kde call later
        return chain
    
#chain=MCMC_MHinGibbs_b(x)

#chain=MCMC_MHinGibbs(x)

#Alg5
"""
Warning this function will throw an error if only this algorithm is run since i have constructed it like the NEville thesis version where it is only proposing new 'a' values so b and eta don't change thus leading to issues with the gaussian kde for those chains.
"""
def MCMC_MIS_a(x,N, ReturnLast=True):
    #MH algorithm
    L = [x]  # output chain
    for _ in range(N):
        u = np.random.uniform()  # uniform draw to test acceptance
        x2 = a_unif_proposal_draw(x1=x)  # draw a single candidate

        LL2 = logposterior(y=y, x=x2) + a_unif_logproposal_distribution(x2=x, x1=x2)
        LL1 = logposterior(y=y, x=x) + a_unif_logproposal_distribution(x2=x2, x1=x)
        LLdifference = LL2 - LL1
        A = np.where(LLdifference >= 0, 0, LLdifference)
        if u <= np.exp(A):  # TODO array version, likely need cleanup
            x = x2
        L.append(x)
    #Hashing out warmup since need to only take last result for most of these algorithms
    #warmup = N // 10
    #L = L[warmup:]  # dump first 10% of the chain
    if ReturnLast:
        state=L[-1]
        return state
    else:
        chain = np.array(L)
        chain = chain.reshape((-1, 3))
        chain = chain.transpose()  # not very meaningful to transpose, just simpler for kde call later
        return chain

#chain=MCMC_MIS_a(x)

def MCMC_MIS(x,N):
    #MH algorithm
    L = [x]  # output chain
    for _ in range(N):
        u = np.random.uniform()  # uniform draw to test acceptance
        x2 = MIS_proposal_draw(x1=x)  # draw a single candidate

        LL2 = logposterior(y=y, x=x2) + MIS_logproposal_distribution(x2=x, x1=x2)
        LL1 = logposterior(y=y, x=x) + MIS_logproposal_distribution(x2=x2, x1=x)
        LLdifference = LL2 - LL1
        A = np.where(LLdifference >= 0, 0, LLdifference)
        if u <= np.exp(A):  # TODO array version, likely need cleanup
            x = x2
        L.append(x)
    warmup = N // 10
    L = L[warmup:]  # dump first 10% of the chain
    chain = np.array(L)
    chain = chain.reshape((-1, 3))
    chain = chain.transpose()  # not very meaningful to transpose, just simpler for kde call later
    return chain

#chain=MCMC_MIS(x)

#Alg6
"""
Warning this function will throw an error if only this algorithm is run since i have constructed it like the NEville thesis version where it is only proposing new 'a' values so b and eta don't change thus leading to issues with the gaussian kde for those chains.
"""
def MCMC_MISinGibbs_a(x,N, ReturnLast=True):
    #MIS-within-Gibbs alg
    L = [x]  # output chain
    for _ in range(N):
        for i in range(len(x[0])):
            if i in [0]:
                u = np.random.uniform()  # uniform draw to test acceptance
                x2 = MIS_a_proposal_draw(x1=x)  # draw a single candidate

                LL2 = logposterior(y=y, x=x2) + MIS_logproposal_distribution(x2=x, x1=x2)
                LL1 = logposterior(y=y, x=x) + MIS_logproposal_distribution(x2=x2, x1=x)
                LLdifference = LL2 - LL1
                A = np.where(LLdifference >= 0, 0, LLdifference)
                if u <= np.exp(A):  # TODO array version, likely need cleanup
                    x = x2
        L.append(x)
    #Hashing out warmup since need to only take last result for most of these algorithms
    #warmup = N // 10
    #L = L[warmup:]  # dump first 10% of the chain
    if ReturnLast:
        state=L[-1]
        return state
    else:
        chain = np.array(L)
        chain = chain.reshape((-1, 3))
        chain = chain.transpose()  # not very meaningful to transpose, just simpler for kde call later
        return chain

#chain=MCMC_MISinGibbs_a(x)

def MCMC_MISinGibbs(x,N):
    #MIS-within-Gibbs alg
    L = [x]  # output chain
    for _ in range(N):
        for i in range(len(x[0])):
            if i in [0]:
                u = np.random.uniform()  # uniform draw to test acceptance
                x2 = MIS_a_proposal_draw(x1=x)  # draw a single candidate

                LL2 = logposterior(y=y, x=x2) + MIS_logproposal_distribution(x2=x, x1=x2)
                LL1 = logposterior(y=y, x=x) + MIS_logproposal_distribution(x2=x2, x1=x)
                LLdifference = LL2 - LL1
                A = np.where(LLdifference >= 0, 0, LLdifference)
                if u <= np.exp(A):  # TODO array version, likely need cleanup
                    x = x2
            if i in [1]:
                u = np.random.uniform()  # uniform draw to test acceptance
                x2 = MIS_b_proposal_draw(x1=x)  # draw a single candidate

                LL2 = logposterior(y=y, x=x2) + MIS_logproposal_distribution(x2=x, x1=x2)
                LL1 = logposterior(y=y, x=x) + MIS_logproposal_distribution(x2=x2, x1=x)
                LLdifference = LL2 - LL1
                A = np.where(LLdifference >= 0, 0, LLdifference)
                if u <= np.exp(A):  # TODO array version, likely need cleanup
                    x = x2
            if i in [2]:
                u = np.random.uniform()  # uniform draw to test acceptance
                x2 = MIS_eta_proposal_draw(x1=x)  # draw a single candidate

                LL2 = logposterior(y=y, x=x2) + MIS_logproposal_distribution(x2=x, x1=x2)
                LL1 = logposterior(y=y, x=x) + MIS_logproposal_distribution(x2=x2, x1=x)
                LLdifference = LL2 - LL1
                A = np.where(LLdifference >= 0, 0, LLdifference)
                if u <= np.exp(A):  # TODO array version, likely need cleanup
                    x = x2
        L.append(x)

    warmup = N // 10
    L = L[warmup:]  # dump first 10% of the chain
    chain = np.array(L)
    chain = chain.reshape((-1, 3))
    chain = chain.transpose()  # not very meaningful to transpose, just simpler for kde call later
    return chain

#chain=MCMC_MISinGibbs(x)

#Alg7
"""
Warning this function will throw an error if only this algorithm is run since i have constructed it like the NEville thesis version where it is only proposing new 'a' values so b and eta don't change thus leading to issues with the gaussian kde for those chains.
"""
def MCMC_pi_kick(x,N, ReturnLast=True):
    L = [x]  # output chain
    for _ in range(N):
        x2=pi_draw(x1=x) # draw candidate

        LL2 = logposterior(y=y, x=x2) #new
        LL1 = logposterior(y=y, x=x) #current
        LLdifference = LL2 - LL1

        if LLdifference>0:
            x=x2
        L.append(x)
    #Hashing out warmup since need to only take last result for most of these algorithms
    #warmup = N // 10
    #L = L[warmup:]  # dump first 10% of the chain
    if ReturnLast:
        state=L[-1]
        return state
    else:
        chain = np.array(L)
        chain = chain.reshape((-1, 3))
        chain = chain.transpose()  # not very meaningful to transpose, just simpler for kde call later
        return chain

#chain=MCMC_pi_kick(x)

def MCMC_MH(x,N):
    #MH algorithm
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
    chain = chain.transpose()  # not very meaningful to transpose, just simpler for kde call later
    return chain

#chain=MCMC_MH(x)

p_alpha=x

for i in range(I[0]): #step 2.2
    #step 2.2i
    p_alpha=MCMC_MIS_a(p_alpha,I[1],ReturnLast=True)
    print(p_alpha)
    print("###step 2.2i done###")
    #step 2.2ii
    p_alpha=MCMC_MISinGibbs_a(p_alpha,I[2],ReturnLast=True)
    print(p_alpha)
    print("###step 2.2ii done###")
    #step 2.2iii
    #p_alpha=Alg4(p_alpha, I[3]) #p_alpha is first p_alpha
    p_alpha=MCMC_MHinGibbs_a(p_alpha, I[3],ReturnLast=True) #p_alpha is second p_alpha
    print(p_alpha)
    print("###step 2.2iii done###")
    #step 2.2iv (and 2.2v)
    p_alpha=MCMC_pi_kick(p_alpha,I[4],ReturnLast=True)
    print(p_alpha)
    print("###step 2.2iv done###")

#p_beta=[0.5,0.5,0.5,p_alpha[3],p_alpha[4],b_est,b_est] #step 2.3
#p_beta=[0.5,0.5,0.5,p_alpha[3],p_alpha[4],0.7,0.7]
p_beta=p_alpha

print("p_beta initial is: {}".format(p_beta))
#step 2.4
p_beta=MCMC_MHinGibbs_b(p_beta, I[5],ReturnLast=True)
print(p_beta)
print("###step 2.4 done###")

#p_zero=[0.5,0.5,0.5,p_beta[3],p_beta[4],p_beta[5],p_beta[6]] #step 2.5
p_zero=p_beta
print("p_zero is: {}".format(p_zero))

#step 2.6
p_zero=MCMC_MHinGibbs(p_zero,I[6], ReturnLast=True)
print(p_zero)
print("###step 2.6 done###")

#p_conv=list(p_zero) #step 2.7
p_conv=p_zero
print("p_conv is: {}".format(p_conv))


###Main Markov Chain Generation###

#Step 3
chain=MCMC_MHinGibbs(p_conv,I[7], ReturnLast=False)

# Data visualisation
# I no longer plot the exact posterior because it's a pain when using loglikelihood

# marginal in a
kernel_a = scipy.stats.gaussian_kde(chain[0])
a_range = np.linspace(-1, 1, 1000)
estimated_pdf_a = kernel_a.evaluate(a_range)  # already normalised

print("a for MAP estimator: ", a_range[np.argmax(estimated_pdf_a)])
print("a for MMSE estimator: ", np.mean(chain[0]))
print("a exact: ", a)
print()

plt.figure()
plt.plot(a_range, estimated_pdf_a, "-", label="Kernel", color="red")
plt.axvline(x=a, label="Exact a", color="blue")
plt.legend()
plt.xlim([0, 1])
#plt.ylim([0, np.round(1.2*np.max([estimated_pdf_a]))])
# plt.show()
plt.savefig("a.png")

plt.figure()
plt.plot(chain[0], color="blue")
plt.axhline(y=a, label="Exact a", color="red")
plt.legend()
#plt.ylim([0, 1])
#plt.xlim([0, np.round(1.2*np.max([estimated_pdf_a]))])
# plt.show()
plt.savefig("a_state.png")


# marginal in b
kernel_b = scipy.stats.gaussian_kde(chain[1])
b_range = np.linspace(-1, 1, 1000)
estimated_pdf_b = kernel_b.evaluate(b_range)  # already normalised

print("b for MAP estimator: ", b_range[np.argmax(estimated_pdf_b)])
print("b for MMSE estimator: ", np.mean(chain[1]))
print("b exact: ", b)
print()

plt.figure()
plt.plot(b_range, estimated_pdf_b, "-", label="Kernel", color="red")
plt.axvline(x=b, label="Exact b", color="blue")
plt.legend()
plt.xlim([-1, 1])
#plt.ylim([0, np.round(1.2*np.max([estimated_pdf_b]))])
# plt.show()
plt.savefig("b.png")

plt.figure()
plt.plot(chain[1], color="blue")
plt.axhline(y=b, label="Exact b", color="red")
plt.legend()
#plt.ylim([-1, 1])
#plt.xlim([0, np.round(1.2*np.max([estimated_pdf_b]))])
# plt.show()
plt.savefig("b_state.png")

# marginal in eta
kernel_eta = scipy.stats.gaussian_kde(chain[2])
eta_range = np.linspace(0, 1, 1000)
estimated_pdf_eta = kernel_eta.evaluate(eta_range)  # already normalised

print("eta for MAP estimator: ", eta_range[np.argmax(estimated_pdf_eta)])
print("eta for MMSE estimator: ", np.mean(chain[2]))
print("eta exact: ", eta)


plt.figure()
plt.plot(eta_range, estimated_pdf_eta, "-", label="Kernel", color="red")
plt.axvline(x=eta, label="Exact eta", color="blue")
plt.legend()
plt.xlim([0, 1])
#plt.ylim([0, np.round(1.2*np.max([estimated_pdf_eta]))])
# plt.show()
plt.savefig("eta.png")

plt.figure()
plt.plot(chain[2], color="blue")
plt.axhline(y=eta, label="Exact eta", color="red")
plt.legend()
#plt.ylim([0, 1])
#plt.xlim([0, np.round(1.2*np.max([estimated_pdf_eta]))])
# plt.show()
plt.savefig("eta_state.png")