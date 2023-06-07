"""
This code will be dedicated to generalising MainCode.py so it isn't specifically built around the toy example.

I have copied over the code from my generalised Aux_Nev code so that it is all in the same file
for completeness.

'a' values seem to deviate strongly after step 2.iii so on step iv it jumps up to worse numbers.
so it is algorithm 7 (the stochastic pi kick search) that starts the big deviation. I think 
the code is functionally doing what it should be, the only thing i haven't debugged is the 
likelihood and the inequality check but the likelihood should surely be working fine since 
it works in the other algs...

So really my main thing is that i think the deviation lies in the inequality criterion, my
hardcoded variant raises runtime errors while my generalised code doesn't so maybe that highlights some troubles. 

Hardcoded likelihoods often wind up rounding to zero, while this version is on the order of -232 and so isn't rounded to zero
smallest value is usually around E-308 so really it can be seen then that there is some rounding errors.
It still seems strange to me that the code that is generating worse estimates for a. What if i modulo 2pi
it, is that valid and will they still be good? They will still be good, so now it is just a case of knowing if that is valid to do...

List for collating runtimes for different scenarios of format with Vlength=100, [paramnums,MCMClength]=runtime

[7,100]=66.27658700942993
[7,1000]=334.0539951324463
[10,100]=115.77437496185303
[10,1000]=713.606169462204
[13,100]=132.8034279346466
[13,1000]=729.9649174213409

Notes for further generalisation is that i am still hardcoding my Voltages and my p_alphas so they need to be sorted, have worked with for my runtime investigation but will need to sort.
"""

import numpy as np
import scipy
from collections import OrderedDict
import time

#####################Generating Data#######################

RANDOM_SEED = 8927 #Seed for numpy random
rng = np.random.default_rng(RANDOM_SEED)

#########################Supporting functions
#Function for removing dictionary elements
def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

#Define formula for Phase shifter unitary, will define just dim=2 for now
def construct_PS(phi):
    mat=np.array(([np.exp(1j*phi/2),0],[0,np.exp(-1j*phi/2)]))
    return mat

def construct_BS(eta):
    mat=np.array(([np.sqrt(eta), 1j*np.sqrt(1-eta)],[1j*np.sqrt(1-eta),np.sqrt(eta)]))
    return mat

top_ket=np.array([1,0])
top_ket.shape=(2,1)

top_bra=np.array([1,0])
top_bra.shape=(1,2)

bottom_ket=np.array([0,1])
bottom_ket.shape=(2,1)

bottom_bra=np.array([0,1])
bottom_bra.shape=(1,2)

from collections import defaultdict

a1_true=0
a2_true=0

b1_true=0.788
b2_true=0.711

eta1_true=0.447
eta2_true=0.548
eta3_true=0.479

#circuit_ordering=[('BS',0.5),('PS',np.pi),('BS',0.49),('PS',-np.pi),('BS',0.51)]
#Phase shifter stores a,b as [a,b], i will need to remember this in future components.
#Additionally i will have to remember that circuit ordering and total ordering is reading from left to right
#circuit_ordering=[('BS',eta1_true),('PS',[a1_true,b1_true]),('BS',eta2_true),('PS',[a2_true,b2_true]),('BS',eta3_true)]
circuit_ordering=[('BS',eta2_true),('PS',[a2_true,b2_true]),('BS',eta3_true),('PS',[a1_true,b2_true]),('BS',eta2_true),('BS',eta1_true),('PS',[a1_true,b1_true])]
circuit = defaultdict(list)
totalorder=[]
for k, v in circuit_ordering:
    circuit[k].append(v)
    totalorder.append(k)

#print(circuit)
#print(totalorder)

#constructU will take calculated phi value from datagen so i can keep constructU generic
#I will build my function like i pass a unitary dictionary to it where it accepts eta values and phi values
#e.g.U_dict={'BS':[BS1,BS2,BS3],'PS':[PS1,PS2]}
def constructU(**kwargs):
    U=np.eye(2)
    BS_counter=0
    PS_counter=0
    for i in range(len(totalorder)):
        if totalorder[i]=='BS':
            print(circuit['BS'][BS_counter])
            U=U@construct_BS(circuit['BS'][BS_counter])
            BS_counter+=1
        if totalorder[i]=='PS':
            print(circuit['PS'][PS_counter])
            U=U@construct_PS(circuit['PS'][PS_counter])
            PS_counter+=1
    return U

#Need a seperate function that takes my p_V dictionary that gets passed to my Likelihood function
#which will have different keys
def constructU_from_p(etas,phis):
    U=np.eye(2)
    #print(etas)
    #print(phis)
    BS_counter=0
    PS_counter=0
    for i in range(len(totalorder)):
        if totalorder[i]=='BS':
            U=U@construct_BS(etas[BS_counter])
            BS_counter+=1
        if totalorder[i]=='PS':
            U=U@construct_PS(phis[PS_counter])
            PS_counter+=1
    return U

Vmax=5
N=100 #Top of page 108 ->N=number of experiments
M=2 #Number of modes
V1=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5 for phase shifter 1
#V1=V1+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
V2=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5 for phase shifter 2
#V2=V2+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
V3=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5 for phase shifter 2


#Voltage array being passed
V=[]
V.append(list(V1))
#print(list(V1))
#print(V)
V.append(list(V2))
#print(list(V2))
#V=V[0]
V.append(list(V3))
#V.append(list(V4))
expanded_dict= circuit.copy()
#print(circuit)
expanded_dict['V'].append(V)
#print(circuit)
#*V should be length same as number of phase shifters
#print(expanded_dict)
"""
but then to generate phi values i need to relate a,b values as well as the voltage across each phase shifter for a given experiment so what may be necessary is another default_dict
like {'BS':[BSarray],'PS':[[a1,b1],[a2,b2]],V:[[V1array],[V2array]]}.
"""
#print(expanded_dict)
def DataGen(InputNumber,poissonian=False, **expanded_dict):
    data=np.empty((N,M))
    C=np.empty(N)

    for i in range(N):
        phis=[]
        for j in range(len(expanded_dict['PS'])):
            #print(expanded_dict['PS'][j][0])
            #print(expanded_dict['PS'][j][1])
            #print(expanded_dict['V'][0][0])
            phis.append(expanded_dict['PS'][j][0]+expanded_dict['PS'][j][1]*expanded_dict['V'][0][j][i]**2)
            #print(phis)
        #_=removekey(expanded_dict,'V')
        U_true=constructU_from_p(expanded_dict['BS'],phis)
        #U_true=constructU(**expanded_dict)        
        P_click1_true=abs(top_bra@U_true@top_ket)**2 #Probability of click in top
        P_click1_true=P_click1_true[0][0]
        P_click2_true=abs(bottom_bra@U_true@top_ket)**2 #Probability of click in bottom
        P_click2_true=P_click2_true[0][0]
        P_true=[P_click1_true,P_click2_true]
        #n=C,p=P,x=array of clicks
        data[i]=scipy.stats.multinomial.rvs(n=InputNumber,p=P_true)
        #Need to add poissonian noise
        if poissonian==True:
            data[i]+=rng.poisson(size=len(data[i]))
        C[i]=np.sum(data[i])

    return data,C

"""
Haven't formally checked DataGen so keep that in mind.
"""

#data,C=DataGen(InputNumber=1000,Voltages=V,poissonian=False)
data,C=DataGen(InputNumber=1000,poissonian=False, **expanded_dict)
#print(np.shape(data))
#print(data) #Correct
#print(C)

"""
The final component of this document then is to create the likelihood python function.
What passes to this is a generic array 'p' of parameter values and V values in the standard case,
so then the thing to think about is what am i actually going to parse to this function. This 
should reflect what i will be using in the maincode which i think will just be another dictionary.
since i seem to get an error when i try to pass two **kwargs (e.g. foo(**p,**V), i will just make use of
a dictionary that has p and V combined)
"""

def Likelihood(**p_V):
    #To be called after data generation
    P=np.empty((N,M))
    prob=np.empty(N)

    for i in range(N):
        phis=[]
        for j in range(len(p_V['a'])):
            #phis.append(expanded_dict['a'][j]+expanded_dict['b'][j]*expanded_dict['V'][j][i]**2)
            phis.append(p_V['a'][j]+p_V['b'][j]*p_V['V'][0][j][i]**2)
        etas=p_V['eta']
        #print(etas)
        #print(phis)
        U=constructU_from_p(etas,phis)
        P_click1=np.abs(top_bra@U@top_ket)**2 #Probability of click in top
        P_click1=P_click1[0][0]
        P_click2=np.abs(bottom_bra@U@top_ket)**2 #Probability of click in bottom
        P_click2=P_click2[0][0]
        P[i]=[P_click1,P_click2]
        #n=C,p=P,x=array of clicks
        prob[i]=np.log(scipy.stats.multinomial.pmf(x=data[i],n=C[i],p=P[i]))
        if np.isinf(prob[i]):
            prob[i]=0 #To bypass -inf ruining likelihood calculations.
    #print(prob)
    logsum=np.sum(prob)
    return logsum

##########################################################################################
#End of Aux Nev component
##########################################################################################

def normal(x,mu,sigma):
    numerator = np.exp((-(x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)
    return numerator/denominator

def uniform(x):
    return 1/(2*np.pi)

def random_coin(p):
    unif = np.random.uniform(0,1)
    if unif<=p:
        return True
    else:
        return False

#eq 4.11: g_i(p',p)=Normal(p_i,sigma_i)
#sigma_i=pi/200 for a, b_est for b, 0.005 for eta

b_est=0.07 
"""
b_est is subject to change and consistency w. Aux_Nev should be certified, additionally sigma_i for b is 
suggested it should be b_est but the actual estimated value is closer to 0.7 so there is confusion there.
"""

eta_sigma=0.005
a_sigma=np.pi/200
b_sigma=b_est #Based around true values from Neville_thesis_8.py
#N_iters=100000

#I=[2,500,50,50,500,100,100,100000] #Determines iteration number for each algorithm call
I=[2,500,50,50,500,100,100,100]

#I[-1]=1 iteration takes 0.08s,10 takes 0.8 so 100,000 should take ~10,000s=~1667min=~27 hours=~1.15 days
#AKA divide I[-1] by 10 to get approx. runtime
# Numba tend to boast an order or two orders of magnitude speedup in general (with caveats ofc)
# 1 order of magnitude:  ~1000s=~16-17mins
# 2 orders of magnitude: ~100s=~1-2 mins

runtime=(I[-1]/10)+60
#print("runtime in seconds is around {}s".format(runtime))
#print("runtime in minutes is around {}min".format(runtime/60))
#print("runtime in hours is around {}hr".format(runtime/(3600)))

###Burn in###

#p_alpha=[0.5,0.5,0.5,0,0,0.5,0.5] #step 2.1
#p_alpha=[0,0] #step 2.1

#p_alpha_list=[('eta',0.5),('a',0),('b',0.5),('eta',0.5),('a',0),('b',0.5),('eta',0.5)]
p_alpha_list=[('eta',0.5),('a',0),('b',0.5),('eta',0.5),('a',0),('b',0.5),('eta',0.5),('eta',0.5),('a',0),('b',0.5)]
p_alpha = defaultdict(list)
for k, v in p_alpha_list:
    p_alpha[k].append(v)

print(p_alpha)

p_alpha['V'].append(V)

"""
Defining Algorithms from thesis. Have done other documents that implement them but i 
shall use those to define these functions which should give a specific output as wanted.
"""

def Alg4_alpha(Niters,**p_alpha):
    """
    Algorithm variant of Algorithm 4 for estimating the alpha model (which involves 
    only learning the a values).
    """
    for n in range(Niters):
        for k,v in p_alpha.items():
            if k == 'a': #If it is a's
                for i in range(len(v)):
                    new_element=np.random.normal(loc=p_alpha['a'][i],scale=a_sigma) #draw random sample from proposal distribution
                    #p_prime=list(p_alpha)
                    p_prime=p_alpha.copy()
                    #p_prime[i]=new_element
                    p_prime['a'][i]=new_element
                    #Likelihood
                    #L1=Likelihood(p_alpha,V1,V2)
                    L1=Likelihood(**p_alpha)
                    #L2=Likelihood(p_prime,V1,V2)
                    L2=Likelihood(**p_prime)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    #P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
                    P1=uniform(p_alpha['a'][i])
                    P2=uniform(p_prime['a'][i])
                    #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                    
                    #Candidates
                    g1= np.random.normal(p_alpha['a'][i],a_sigma)
                    g2=np.random.normal(p_prime['a'][i],a_sigma)
                    numerator=L1*P1*g1
                    denominator=L2*P2*g2
                    elem=numerator/denominator
                    T=min(1,elem)
                    move_prob=random_coin(T)
                    if move_prob:
                        p_alpha=p_prime

    return p_alpha

def Alg4_beta(Niters, **p_beta):
    """
    Algorithm variant of Algorithm 4 for estimating the beta model (which involves 
    learning the a and b values).
    """
    for n in range(Niters):
        for k,v in p_beta.items():
            if k == 'a': #If it is a's
                for i in range(len(v)):
                    new_element=np.random.normal(loc=p_beta['a'][i],scale=a_sigma) #draw random sample from proposal distribution
                    #p_prime=list(p_alpha)
                    p_prime=p_alpha.copy()
                    #p_prime[i]=new_element
                    p_prime['a'][i]=new_element
                    #Likelihood
                    #L1=Likelihood(p_alpha,V1,V2)
                    L1=Likelihood(**p_beta)
                    #L2=Likelihood(p_prime,V1,V2)
                    L2=Likelihood(**p_prime)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    #P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
                    P1=uniform(p_beta['a'][i])
                    P2=uniform(p_prime['a'][i])
                    #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                    
                    #Candidates
                    g1= np.random.normal(p_beta['a'][i],a_sigma)
                    g2=np.random.normal(p_prime['a'][i],a_sigma)
                    numerator=L1*P1*g1
                    denominator=L2*P2*g2
                    elem=numerator/denominator
                    T=min(1,elem)
                    move_prob=random_coin(T)
                    if move_prob:
                        p_beta=p_prime
            if k == 'b': #If it is b's
                for i in range(len(v)):
                    new_element=np.random.normal(loc=p_beta['b'][i],scale=b_sigma) #draw random sample from proposal distribution
                    #p_prime=list(p_alpha)
                    p_prime=p_alpha.copy()
                    #p_prime[i]=new_element
                    p_prime['b'][i]=new_element
                    #Likelihood
                    #L1=Likelihood(p_alpha,V1,V2)
                    L1=Likelihood(**p_alpha)
                    #L2=Likelihood(p_prime,V1,V2)
                    L2=Likelihood(**p_prime)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    #P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
                    P1=uniform(p_beta['b'][i])
                    P2=uniform(p_prime['b'][i])
                    #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                    
                    #Candidates
                    g1= np.random.normal(p_beta['b'][i],b_sigma)
                    g2=np.random.normal(p_prime['b'][i],b_sigma)
                    numerator=L1*P1*g1
                    denominator=L2*P2*g2
                    elem=numerator/denominator
                    T=min(1,elem)
                    move_prob=random_coin(T)
                    if move_prob:
                        p_beta=p_prime
    return p_beta

#In this instantiation MCMC will be an array of dictionaries (with the V values popped out)
def Alg4(Niters,Markov=False,ReturnAll=False,**p):
    """
    This Algorithm is the Metropolis-Hastings within Gibbs sampling algorithm that is 
    described on the middle of page 94 in Alex Neville's thesis.
    """
    if Markov: #i.e. if Markov==True
        MCMC=Alg4(Niters, Markov=False, ReturnAll=True,**p)
        return MCMC
    else: #If Markov==False
        MCMC=[]
        for n in range(Niters):
            for k,v in p_beta.items():
                if k == 'eta': #If it is eta's
                    for i in range(len(v)):
                        new_element=np.random.normal(loc=p['eta'][i],scale=eta_sigma) #draw random sample from proposal distribution
                        #p_prime=list(p_alpha)
                        p_prime=p.copy()
                        #p_prime[i]=new_element
                        p_prime['eta'][i]=new_element
                        #Likelihood
                        #L1=Likelihood(p_alpha,V1,V2)
                        L1=Likelihood(**p)
                        #L2=Likelihood(p_prime,V1,V2)
                        L2=Likelihood(**p_prime)
                        #Priors
                        #eta: mu=0.5,sigma=0.05
                        #a: uniform so N/A
                        #b: mu=0.7,sigma=0.07
                        #P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
                        P1=uniform(p['eta'][i])
                        P2=uniform(p_prime['eta'][i])
                        #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                        
                        #Candidates
                        g1= np.random.normal(p['eta'][i],eta_sigma)
                        g2=np.random.normal(p_prime['eta'][i],eta_sigma)
                        numerator=L1*P1*g1
                        denominator=L2*P2*g2
                        elem=numerator/denominator
                        T=min(1,elem)
                        move_prob=random_coin(T)
                        if move_prob:
                            p=p_prime
                if k == 'a': #If it is a's
                    for i in range(len(v)):
                        new_element=np.random.normal(loc=p['a'][i],scale=a_sigma) #draw random sample from proposal distribution
                        #p_prime=list(p_alpha)
                        p_prime=p.copy()
                        #p_prime[i]=new_element
                        p_prime['a'][i]=new_element
                        #Likelihood
                        #L1=Likelihood(p_alpha,V1,V2)
                        L1=Likelihood(**p)
                        #L2=Likelihood(p_prime,V1,V2)
                        L2=Likelihood(**p_prime)
                        #Priors
                        #eta: mu=0.5,sigma=0.05
                        #a: uniform so N/A
                        #b: mu=0.7,sigma=0.07
                        #P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
                        P1=uniform(p['a'][i])
                        P2=uniform(p_prime['a'][i])
                        #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                        
                        #Candidates
                        g1= np.random.normal(p['a'][i],a_sigma)
                        g2=np.random.normal(p_prime['a'][i],a_sigma)
                        numerator=L1*P1*g1
                        denominator=L2*P2*g2
                        elem=numerator/denominator
                        T=min(1,elem)
                        move_prob=random_coin(T)
                        if move_prob:
                            p=p_prime
                if k == 'b': #If it is b's
                    for i in range(len(v)):
                        new_element=np.random.normal(loc=p['b'][i],scale=b_sigma) #draw random sample from proposal distribution
                        #p_prime=list(p_alpha)
                        p_prime=p_alpha.copy()
                        #p_prime[i]=new_element
                        p_prime['b'][i]=new_element
                        #Likelihood
                        #L1=Likelihood(p_alpha,V1,V2)
                        L1=Likelihood(**p)
                        #L2=Likelihood(p_prime,V1,V2)
                        L2=Likelihood(**p_prime)
                        #Priors
                        #eta: mu=0.5,sigma=0.05
                        #a: uniform so N/A
                        #b: mu=0.7,sigma=0.07
                        #P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
                        P1=uniform(p['b'][i])
                        P2=uniform(p_prime['b'][i])
                        #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                        
                        #Candidates
                        g1= np.random.normal(p['b'][i],b_sigma)
                        g2=np.random.normal(p_prime['b'][i],b_sigma)
                        numerator=L1*P1*g1
                        denominator=L2*P2*g2
                        elem=numerator/denominator
                        T=min(1,elem)
                        move_prob=random_coin(T)
                        if move_prob:
                            p=p_prime
            if ReturnAll:
                MCMC.append(removekey(p,'V'))

        if ReturnAll:
            return MCMC
        else:
            return p
        
def Alg5(Niters,**p_alpha):
    """
    This Algorithm is the Metropolised Independence Sampling (MIS) algorithm that is
    described at the top of page 98 in Alex Neville's thesis.
    """
    for n in range(Niters):
        #print(p_alpha)
        p_prime=p_alpha.copy() #Need to use list, lesson learned after 4 years of coding
        #print(p_prime)
        new=[np.random.uniform(low=-np.pi,high=np.pi) for i in range(len(p_alpha['a']))]
        p_prime['a']=new
        #Likelihood
        L1=Likelihood(**p_alpha)
        L2=Likelihood(**p_prime)
        #Priors
        #eta: mu=0.5,sigma=0.05
        #a: uniform so N/A
        #b: mu=0.7,sigma=0.07
        #P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
        #P1=uniform(p_alpha[3])*uniform(p_alpha[4])
        P1=1
        for i in range(len(p_alpha['a'])):
            P1*=uniform(p_alpha['a'][i])
        #P2= normal(p_prime[0],0.5,0.05)* normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
        #P2=uniform(p_prime[3])*uniform(p_prime[4])
        P2=1
        for i in range(len(p_prime['a'])):
            P2*=uniform(p_prime['a'][i])
        elem=(np.exp(L1)*P1)/(np.exp(L2)*P2)
        T=min(1,elem)
        move_prob=random_coin(T)
        if move_prob:
            p_alpha=p_prime
    return p_alpha

def Alg6(Niters,**p_alpha):
    """
    This Algorithm is the Metropolised Independence Sampling (MIS) within Gibbs algorithm 
    that is described in the middle of page 98 in Alex Neville's thesis.
    """
    for n in range(Niters):
        for k,v in p_alpha.items():
            if k == 'a': #If it is a's
                for i in range(len(v)):
                    new_element=np.random.uniform(low=-np.pi,high=np.pi)
                    p_prime=p_alpha.copy()
                    p_prime['a'][i]=new_element
                    #Likelihood
                    L1=Likelihood(**p_alpha)
                    L2=Likelihood(**p_prime)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    #P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
                    P1=uniform(p_alpha['a'][i])
                    #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                    P2=uniform(p_prime['a'][i])
                    numerator=L1*P1
                    denominator=L2*P2
                    elem=numerator/denominator
                    T=min(1,elem)
                    move_prob=random_coin(T)
                    if move_prob:
                        p_alpha=p_prime
    return p_alpha

def Alg7(Niters, **p_alpha):
    """
    This Algorithm is the stochastic π kick search algorithm that is described 
    at the bottom of page 98 in Alex Neville's thesis.
    """
    for n in range(Niters):
        x=np.random.choice(3,len(p_alpha['a']))
        q=[(x[i]-1)*np.pi for i in range(len(x))]
        #print(q)
        #Likelihood
        test=p_alpha.copy()
        #print(test['a'])
        #test[3]+=q[0]
        for i in range(len(test['a'])):
            test['a'][i]+=q[i]
        #print(test['a'])
        #test[4]+=q[1]
        
        L1=Likelihood(**test)
        L2=Likelihood(**p_alpha)
        #Priors
        #eta: mu=0.5,sigma=0.05
        #a: uniform so N/A
        #b: mu=0.7,sigma=0.07
        #P1= normal(test[0],0.5,0.05)*normal(test[1],0.5,0.05)*normal(test[2],0.5,0.05)*uniform(test[3])*uniform(test[4])*normal(test[5],0.7,0.07)*normal(test[6],0.7,0.07) #Prior for p+q
        #P1=uniform(test[3])*uniform(test[4])
        P1=1
        for i in range(len(test['a'])):
            P1*=uniform(test['a'][i])
        #print(P1)
        #P2= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
        #P2=uniform(p_alpha[3])*uniform(p_alpha[4])
        P2=1
        for i in range(len(p_alpha['a'])):
            P2*=uniform(p_alpha['a'][i])
        #print(P2)
        #print(np.exp(L1))
        #print(P1)
        #print(np.exp(L2))
        #print(P2)
        if (np.exp(L1)*P1)>(np.exp(L2)*P2):
            p_alpha=test
    return p_alpha

#Main code bulk

start=time.time()

for i in range(I[0]): #step 2.2
    #step 2.2i
    p_alpha=Alg5(I[1],**p_alpha)
    #print(p_alpha)
    #print("###step 2.2i done###")
    #step 2.2ii
    p_alpha=Alg6(I[2],**p_alpha)
    #print(p_alpha)
    #print("###step 2.2ii done###")
    #step 2.2iii
    #p_alpha=Alg4(p_alpha, I[3]) #p_alpha is first p_alpha
    p_alpha=Alg4_alpha(I[3],**p_alpha) #p_alpha is second p_alpha
    #print(p_alpha)
    #print("###step 2.2iii done###")
    #step 2.2iv (and 2.2v)
    p_alpha=Alg7(I[4],**p_alpha)
    #print(p_alpha)
    #print("###step 2.2iv done###")

#p_beta=[0.5,0.5,0.5,p_alpha[3],p_alpha[4],b_est,b_est] #step 2.3
#p_beta=[0.5,0.5,0.5,p_alpha[3],p_alpha[4],0.7,0.7]
p_beta=p_alpha.copy()
for i in range(len(p_beta['eta'])):
    p_beta['eta'][i]=0.5
for i in range(len(p_beta['b'])):
    p_beta['b'][i]=0.7



#print("p_beta initial is: {}".format(p_beta))
#step 2.4
p_beta=Alg4_beta(I[5],**p_beta)
#print(p_beta)
#print("###step 2.4 done###")

#p_zero=[0.5,0.5,0.5,p_beta[3],p_beta[4],p_beta[5],p_beta[6]] #step 2.5
p_zero=p_beta.copy()
for i in range(len(p_zero['eta'])):
    p_zero['eta'][i]=0.5
#print("p_zero is: {}".format(p_zero))
#step 2.6
p_zero=Alg4(I[6], Markov=False,**p_zero)
#print(p_zero)
#print("###step 2.6 done###")

p_conv=p_zero.copy() #step 2.7
#print("p_conv is: {}".format(p_conv))

###Main Markov Chain Generation###

#Step 3
chain=Alg4(I[7], Markov=True,**p_conv)

end=time.time()

print("runtime is {}".format(end-start))
###Parameter estimation###

"""
I imagine this chain object should contain all the values for each parameter at each 
markov chain state number (i.e. I[7] by 7 matrix).
To start with i shall just generate the markov chain plot and comment out the histogram 
plot and polish can later be applied to get the standard plot with the smoothed histogram in the left column and
markov state number plot in the right column with a Plot() function.
"""
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

#names=["eta1","eta2","eta3","a1","a2","b1","b2"]

#difficulty in setting names since i don't think i can use formatting to set a variable name :(
#so for now i will leave it

"""
Haven't properly checked the below code block, but the intention is to unpack the array of
default dictionaries into a numpy matrix
"""
"""
#This block is throwing errors so i will correct it at some point
z=[]
for i in range(len(chain)):
    dictionary=chain[i]
    blank=[]
    for k,v in dictionary:
        for l in range(len(v)):
            blank.append(dictionary[k][l])
    z.append(blank)

z=np.array(z)
"""

def Plot(chain): #Chain should contain all necessary markov chain data
    """
    Custom plot function to generate the standard traceplot format where there is a smoothed
    histogram for each parameter in the left hand column and the markov chain plot in the right hand column.
    e.g. https://python.arviz.org/en/stable/examples/plot_trace.html
    """
    fig,axs=plt.subplots(len(p_conv),2,constrained_layout=True) #Can use sharex,sharey for further polish if wanted
    for i in range(len(p_conv)):
        #histogram
        #axs[i,0].hist(chain[:,i],bins=30)
        #smoothed kde with scott's rule bandwidth selection (bw selection is important consideration in kde)
        eval_points = np.linspace(np.min(chain[:,i]), np.max(chain[:,i]),len(chain[:,i]))
        kde=gaussian_kde(chain[:,i])
        evaluated=kde.evaluate(eval_points)
        evaluated/=sum(evaluated) #For normalisation
        axs[i,0].plot(eval_points,evaluated)
        #Add axs polish like axes labelling
        #axs[i,0].set_ylabel(str(names[i])) #Add label to show which parameter is which
        axs[i,1].plot(chain[:,i])
        axs[i,1].set_xlabel("Markov chain State Number") #Aid understanding of Markov chain plot
    #fig.tight_layout()
    plt.show()

#print(chain)
"""
chain contains an array of default dictionaries so i sgould figure out how to handle this output.
"""
"""
chain=np.array(chain)
Plot(chain)

for i in range(len(p_conv)): #step 4
    par_array=chain[:,i]
    #Plot markov chain plot
    #print(names[i])
    #plt.plot(par_array)
    print("Mean is {}".format(np.mean(par_array)))
    print("Standard deviation is {}".format(np.std(par_array)))
"""