"""
Code for replicating fig 4.5 in Alex Neville thesis.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

wv=800*1E-9 #Wavelength of light in fig 4.5, =0.8 micrometers

stds=np.array([0.05,0.1,0.2,0.4,0.5,1])*1E-6
y=np.zeros((len(stds),100000))

for i in range(len(stds)):
    x=norm.rvs(loc=0,scale=stds[i],size=100000)
    for j in range(len(x)):
        y[i][j]=(x[j]/(wv))*2*np.pi
        
HIST_BINS = np.linspace(-np.pi, np.pi, 60)
plt.hist(y[0][:],HIST_BINS,density=True,histtype='bar', ec='black')
plt.hist(y[1][:],HIST_BINS,density=True,histtype='bar', ec='black')
plt.hist(y[2][:],HIST_BINS,density=True,histtype='bar', ec='black')
plt.hist(y[3][:],HIST_BINS,density=True,histtype='bar', ec='black')
plt.hist(y[4][:],HIST_BINS,density=True,histtype='bar', ec='black')
plt.hist(y[5][:],HIST_BINS,density=True,histtype='bar', ec='black')
plt.show()

"""
Figure has been successfully replicated, showing that increasing standard 
deviation tends the phase shift from a normal distribution to a uniform 
distribution.
"""