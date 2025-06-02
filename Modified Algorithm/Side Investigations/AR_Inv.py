import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(123)
T = 1000
alpha = 0.9
beta = 0.98
sigma_u = 0.1
sigma_v = 0.05
h = np.zeros(T)
y = np.zeros(T)
y1 = np.zeros(T)

h[0] = np.random.normal(0, sigma_u / np.sqrt(1 - beta ** 2))
y[0] = np.random.normal(0, np.exp(h[0] / 2))
y1[0] = y[0]
 
for t in range(1, T):
    h[t] = beta * h[t - 1] + np.random.normal(0, sigma_u)
    y[t] = alpha * y[t - 1] + np.exp(h[t] / 2) * np.random.normal(0, sigma_v)
    y1[t] = alpha * y1[t - 1] + np.random.normal(0, sigma_v)

import pymc as pm

with pm.Model() as model:
    # Priors for the parameters
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta = pm.Normal("beta", mu=0, sigma=1)
    sigma_u = pm.HalfNormal('sigma_u', sigma=1)
    sigma_v = pm.HalfNormal('sigma_v', sigma=1)
 
    # Latent volatility process
    h = pm.AR('h', rho=beta, tau=sigma_u**-2, shape=T)
 
    # Observed data
    y_obs = pm.Normal('y_obs', mu=alpha * y[:-1], sigma=pm.math.exp(h[1:] / 2) * sigma_v, observed=y[1:])
 
    # Perform MCMC sampling
    trace = pm.sample(500, tune=1000, chains=2)

import arviz as az

print(trace.posterior['h'])
az.plot_trace(trace,var_names=['h'])




