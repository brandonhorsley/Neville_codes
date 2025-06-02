"""
Algorithm 1 in Alex Neville thesis. Rejection sampling. Page 35 in thesis.

For i = 1 to i = N
1. Sample x_(i) ~ q(x) and u ~ unif(0, 1).
2. If u < p(x^(i))/Mq(x^(i)):
      Accept x^(i) and set i to i + 1.
   Else:
      Reject x^(i)
"""

#https://jaketae.github.io/study/rejection-sampling/

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Target distribution, simple gaussian mixture
def p(x):
    return norm.pdf(x, loc=30, scale=10) + norm.pdf(x, loc=80, scale=20)

#Envelope distribution
def q(x):
    return norm.pdf(x, loc=50, scale=30)

#Compare plots
x = np.arange(-50, 151)
fig, ax = plt.subplots()
ax.plot(x, p(x), label=r"$p(x)$")
ax.plot(x, q(x), label=r"$q(x)$")
plt.legend()
plt.show()

#Normalisation constant
M = max(p(x) / q(x))

#Show comparison
fig, ax = plt.subplots()
ax.plot(x, p(x), label=r"$p(x)$")
ax.plot(x, M * q(x), label=r"$k \cdot q(x)$")
plt.legend()
plt.show()

def sample(size):
    xs = np.random.normal(50, 30, size=size) #q
    cs = np.random.uniform(0, 1, size=size) #u
    mask = p(xs) / (M * q(xs)) > cs #Criterion
    return xs[mask] 

samples = sample(10000)
sns.distplot(samples)
plt.show()