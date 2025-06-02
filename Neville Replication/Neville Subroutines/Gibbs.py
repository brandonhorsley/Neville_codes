"""
Algorithm 3 from Alex Neville thesis. Gibbs sampling algorithm. Bottom of page 40.

1. Pick an initial state vec(x)^(0)
2. For i = 0 to i = N-1
   2.1. Sample x'_1 ~ p(x'_1 | x_2,...,x_n)
   2.2. Sample x'_2 ~ p(x'_2 | x_1,x_3,...,x_n)
   2.j. Sample x'_j ~ p(x'_1 | x_1,...,x_j-1,x_j+1,x_n)
   2.n. Sample x'_n ~ p(x'_n | x_1,...,x_n-1)
   2.n+1. Set vec(x)^(i+1) = (x_1,...x_n)
"""
#https://mr-easy.github.io/2020-05-21-implementing-gibbs-sampling-in-python/

"""

#From:
#https://github.com/ritvikmath/YouTubeVideoCode/blob/main/Gibbs%20Sampling%20Code.ipynb

import numpy as np

samples = {'x': [1], 'y': [-1]}

num_samples = 10000

for _ in range(num_samples):
    curr_y = samples['y'][-1]
    new_x = np.random.normal(curr_y/2, np.sqrt(3/4))
    new_y = np.random.normal(new_x/2, np.sqrt(3/4))
    samples['x'].append(new_x)
    samples['y'].append(new_y)
"""

#From:
#https://towardsdatascience.com/gibbs-sampling-8e4844560ae5

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()
#Posterior we want to sample from
f = lambda x, y: np.exp(-(x*x*y*y+x*x+y*y-8*x-8*y)/2.)

#Plot target distribution
xx = np.linspace(-1, 8, 100)
yy = np.linspace(-1, 8, 100)
xg,yg = np.meshgrid(xx, yy)
z = f(xg.ravel(), yg.ravel())
z2 = z.reshape(xg.shape)
plt.contourf(xg, yg, z2, cmap='BrBG')

#Setting initial values
N = 50000 #N of iterations
x = np.zeros(N+1) #set empty array for X and Y values to be filled by Gibbs protocol
y = np.zeros(N+1)
x[0] = 1. #Initial value for x
y[0] = 6. #Initial value for y
sig = lambda z, i: np.sqrt(1./(1.+z[i]*z[i])) #Define mu since conditional is a normal distribution
mu = lambda z, i: 4./(1.+z[i]*z[i]) #Define sigma since conditional is a normal distribution

#Gibbs protocol
for i in range(1, N, 2):
    #For x
    sig_x = sig(y, i-1)
    mu_x = mu(y, i-1)
    x[i] = np.random.normal(mu_x, sig_x) #Conditional is a normal distribution
    y[i] = y[i-1]
    #For y
    sig_y = sig(x, i)
    mu_y = mu(x, i)
    y[i+1] = np.random.normal(mu_y, sig_y) #Conditional is a normal distribution
    x[i+1] = x[i]

#Plot histograms
plt.hist(x, bins=50)
plt.hist(y, bins=50)

#Plot probability distribution
plt.contourf(xg, yg, z2, alpha=0.8, cmap='BrBG')
plt.plot(x[::10],y[::10], '.', alpha=0.1)
plt.plot(x[:300],y[:300], c='r', alpha=0.3, lw=1)

