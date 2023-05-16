"""
Code to implement the main parameter estimation procedure from Alex Neville's code which is described in 
the entirety of chapter 4.3.4.

I will keep it specific to the toy example case first and then generalise later.
"""

import numpy as np
from Aux_Nev import *

#Data generation (step 1) stuff is all contained in Aux_Nev so i will leave it that way for now for readability 
#but if i need differed functionality then i can bring the code into this code document.

###Preliminaries###

def normal(x,mu,sigma):
    numerator = np.exp((-(x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)
    return numerator/denominator

def uniform(x):
    return 1/(2*np.pi)

def random_coin(p):
    unif = np.random.uniform(0,1)
    if unif>=p:
        return False
    else:
        return True

#Define initial state p=(eta's,a's,b's)
#p=[0.5,0.5,0.5,0,0,0.5,0.5]
#V=2.5

#p={"eta1": 0.5, "eta2": 0.5, "eta3": 0.5, "a1": 0, "a2": 0, "b1": 0.5, "b1": 0.5} #initial vec(p)

#eq 4.11: g_i(p',p)=Normal(p_i,sigma_i)
#sigma_i=pi/200 for a, b_est for b, 0.005 for eta
eta_sigma=0.005
a_sigma=np.pi/200
b_sigma=0.75 #Based around true values from Neville_thesis_8.py
N_iters=100000

eta1_arr=[p[0]]
eta2_arr=[p[1]]
eta3_arr=[p[2]]
a1_arr=[p[3]]
a2_arr=[p[4]]
b1_arr=[p[5]]
b2_arr=[p[6]]

I=[2,500,50,50,500,100,100,100000] #Determines iteration number for each algorithm call
b_est=0.7 #Subject to change and consistency w. Aux_Nev should be certified

###Burn in###

p_alpha=[0.5,0.5,0.5,0,0,0.5,0.5] #has both a's being 0 as needed

"""
Defining Algorithms from thesis. Have done other documents that implement them but i 
shall use those to define these functions which should give a specific output as wanted.

Starting w/ stub code for now though.
"""
#Procedure seems to refer to modifying this algorithm to target a specific p, perhaps i may need a target arg.
#Either that or define separate funcs for each variant but i don't think that will be necessary.
def Alg4(p,Markov=False):
    pass

def Alg5(p_alpha):
    pass

def Alg6(p_alpha):
    pass

def Alg7(p_alpha):
    pass

for i in range(I[0]): #step 2.2
    for j in range(I[1]): #step 2.2i
        p_alpha=Alg5(p_alpha)
    for j in range(I[2]): #step 2.2ii
        p_alpha=Alg6(p_alpha)
    for j in range(I[3]): #step 2.2iii
        p_alpha=Alg4(p_alpha, Markov=False)
    for j in range(I[4]): #step 2.2iv (and 2.2v)
        p_alpha=Alg7(p_alpha)

p_beta=[0.5,0.5,0.5,p_alpha[3],p_alpha[4],b_est,b_est] #step 2.3

for i in range(I[5]): #step 2.4
    p_beta=Alg4(p_beta, Markov=False)

p_zero=[0.5,0.5,0.5,p_beta[3],p_beta[4],p_beta[5],p_beta[6]] #step 2.5

for i in range(I[6]): #step 2.6
    p_zero=Alg4(p_zero,Markov=False)

p_conv=p_zero #step 2.7

###Main Markov Chain Generation###

for i in range(I[7]): #Step 3
    chain=Alg4(p_conv,Markov=True)

###Parameter estimation###
#step 4

"""
I imagine this chain object should contain all the values for each parameter at each 
markov chain state number (i.e. I[7] by 7 matrix).
To start with i shall just generate the markov chain plot and comment out the histogram 
plot and polish can later be applied to get the standard plot with the smoothed histogram in the left column and
markov state number plot in the right column with a Plot() function.
"""

names=["eta1","eta2","eta3","a1","a2","b1","b2"]

for i in range(len(p_conv)):
    par_array=chain[:,i]
    #Plot markov chain plot
    print(names[i])
    plt.plot(par_array)
    print("Mean is {}".format(np.mean(par_array)))
    print("Standard deviation is {}".format(np.std(par_array)))

