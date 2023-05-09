"""
Algorithm 2 in Alex Neville thesis. Metropolis-Hastings algorithm. Middle of page 38.

1. Pick an initial state x^(0)
2. For i = 0 to i = N-1
   2.1. Sample u ~ unif (0, 1)
   2.2. Sample x' ~ g(x'|x^(i))
   2.3. If u < A(x'|x^(i))
           x^(i+1) = x'
        else:
           x^(i+1) = x^(i)
"""

#g should be normal distribution centred on x^(i)?
#g is just the markov property since it is probability of hopping from vec(p) to vec(p')

#https://towardsdatascience.com/bayesian-statistics-metropolis-hastings-from-scratch-in-python-c3b10cc4382d
#https://towardsdatascience.com/mcmc-intuition-for-everyone-5ae79fff22b1
#https://github.com/ritvikmath/YouTubeVideoCode/blob/main/MCMC%20Experiments.ipynb

import numpy as np
#import scipy.stats as sp
import matplotlib.pyplot as plt

def normal(x,mu,sigma):
    numerator = np.exp((-(x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)
    return numerator/denominator

def random_coin(p):
    unif = np.random.uniform(0,1)
    if unif>=p:
        return False
    else:
        return True
    
def gaussian_mcmc(hops,mu,sigma):
    states = []
    burn_in = int(hops*0.2)
    current = np.random.uniform(-5*sigma+mu,5*sigma+mu)
    for i in range(hops):
        states.append(current)
        movement = np.random.uniform(-5*sigma+mu,5*sigma+mu)
        
        curr_prob = normal(x=current,mu=mu,sigma=sigma)
        move_prob = normal(x=movement,mu=mu,sigma=sigma)
        
        acceptance = min(move_prob/curr_prob,1)
        if random_coin(acceptance):
            current = movement
    return states[burn_in:]
    
lines = np.linspace(-3,3,1000) #x array
normal_curve = [normal(l,mu=0,sigma=1) for l in lines] #True plot
dist = gaussian_mcmc(100_000,mu=0,sigma=1) #Sampling
plt.hist(dist,normed=1,bins=20) #Showing sampling
plt.plot(lines,normal_curve) #Showing true plot

