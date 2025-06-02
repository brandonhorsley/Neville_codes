"""
This code will be dedicated to generalising MainCode.py so it isn't specifically built around the toy 
example. This is still limited to two modes though.

The technique employed for this generalised variant is to make use of python's 'default dictionaries' 
which essentially permits arrays as dictionary values for a fixed dictionary key, thus i can cleverly 
index these arrays for specific parameter values.

I have copied over the code from my generalised Aux_Nev code so that it is all in the same file
for completeness.

Note also that this code doesn't have some of the troubleshooting resolutions from MainCode.py 
like imposing proper bounds on parameter values.

Reference:
Alg 4's are MH within Gibbs
Alg 5 is MIS
Alg 6 is MIS within Gibbs
Alg 7 is stochastic pi kick search
"""

import numpy as np
import scipy
import time

#####################Generating Data#######################

RANDOM_SEED = 8927 #Seed for numpy random
rng = np.random.default_rng(RANDOM_SEED)

#########################Supporting functions
#Function for removing dictionary elements
#This will be used to remove the voltage bit of my expanded dictionary later on 
# so that it contains just the dictionary of parameters.
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

#Phase shifter stores a,b as [a,b], i will need to remember this in future components.
#Additionally i will have to remember that circuit ordering and total ordering is reading from left to right

#circuit ordering reads circuit type left to right, circuit element as key and then true value as the dictionary 
# value.The below code block converts this into an array to record the ordering of the circuits and the circuit 
# ordering into defaultdict called circuit.
#circuit_ordering=[('BS',eta1_true),('PS',[a1_true,b1_true]),('BS',eta2_true),('PS',[a2_true,b2_true]),('BS',eta3_true)]
circuit_ordering=[('BS',eta1_true),('PS',[a1_true,b1_true])]
circuit = defaultdict(list)
totalorder=[]
for k, v in circuit_ordering:
    circuit[k].append(v)
    totalorder.append(k)

print(circuit)
print(totalorder)

#Need a seperate function that takes my p_V dictionary that gets passed to my Likelihood function
#which will have different keys
#Have to read totalorder backwards for construction of the unitary

def constructU_from_p(etas,phis):
    U=np.eye(2)
    #print(etas)
    #print(phis)
    BS_counter=-1
    PS_counter=-1
    for i in range(len(totalorder)):
        if totalorder[-i]=='BS':
            U=U@construct_BS(etas[BS_counter])
            BS_counter-=1
        if totalorder[-i]=='PS':
            U=U@construct_PS(phis[PS_counter])
            PS_counter-=1
    return U

Vmax=5
N=100 #Top of page 108 ->N=number of experiments
M=2 #Number of modes

#Since each phase shifter has its own set of voltages applied to each other the block of code below makes one array for each phase shifter.
V=[]
for _ in range(len(circuit['PS'])):
    Velem=V1=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5 for phase shifter
    #Velem=Velem+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
    V.append(list(Velem))

expanded_dict= circuit.copy()
expanded_dict['V'].append(V) #exppanded_dict has circuit parameters and the voltages as another key

"""
but then to generate phi values i need to relate a,b values as well as the voltage across each phase shifter for a given experiment so what may be necessary is another default_dict
like {'BS':[BSarray],'PS':[[a1,b1],[a2,b2]],V:[[V1array],[V2array]]}.
"""

def DataGen(InputNumber,poissonian=False, **expanded_dict):
    data=np.empty((N,M))
    C=np.empty(N)

    for i in range(N):
        phis=[]
        for j in range(len(expanded_dict['PS'])):
            phis.append(expanded_dict['PS'][j][0]+expanded_dict['PS'][j][1]*expanded_dict['V'][0][j][i]**2)
        U_true=constructU_from_p(expanded_dict['BS'],phis)        
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


data,C=DataGen(InputNumber=1000,poissonian=False, **expanded_dict)
#print(np.shape(data))
#print(data) #Correct
#print(C)

def Likelihood(**p_V):
    #To be called after data generation
    P=np.empty((N,M))
    prob=np.empty(N)

    for i in range(N):
        phis=[]
        for j in range(len(p_V['a'])):
            phis.append(p_V['a'][j]+p_V['b'][j]*p_V['V'][0][j][i]**2)
        etas=p_V['eta']
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
    logsum=np.sum(prob)
    return logsum

##########################################################################################
#End of Aux Nev component
##########################################################################################

def normal(x,mu,sigma):
    numerator = np.exp((-(x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)
    return numerator/denominator

def uniform(p,lower=-np.pi, upper=np.pi):
    if p>=lower and p<=upper:
        return 1/(upper-lower)
    else:
        return 0

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

#I=[2,500,50,50,500,100,100,100_000] #Determines iteration number for each algorithm call, 100_000 allowed since python 3.6
I=[2,500,50,50,500,100,100,10_000]


###Burn in###
#creating p_alpha
p_alpha_list=[]
for elem in totalorder:
    if elem=='BS':
        newterm=('eta',0.5)
        p_alpha_list.append(newterm)
    if elem=='PS':
        newterm1=('a',0)
        newterm2=('b',0.5)
        p_alpha_list.append(newterm1)
        p_alpha_list.append(newterm2)

p_alpha = defaultdict(list)
for k, v in p_alpha_list:
    p_alpha[k].append(v)

p_alpha['V'].append(V) #putting V in p_alpha


def Alg4_alpha(Niters,**p_alpha):
    """
    Algorithm variant of Algorithm 4 for estimating the alpha model (which involves 
    only learning the a values).
    """
    for n in range(Niters):
        for k,v in p_alpha.items():
            if k == 'a': #If it is a's
                for i in range(len(v)):
                    #new_element=(np.random.normal(loc=p_alpha['a'][i],scale=a_sigma))%(2*np.pi) #draw random sample from proposal distribution
                    new_element=np.random.normal(loc=p_alpha['a'][i],scale=a_sigma) #draw random sample from proposal distribution
                    p_prime=p_alpha.copy()
                    p_prime['a'][i]=new_element #new proposed state
                    #Likelihood
                    L1=Likelihood(**p_alpha)
                    L2=Likelihood(**p_prime)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    P1=uniform(p_alpha['a'][i])
                    P2=uniform(p_prime['a'][i])
                    #Candidates
                    g1= np.random.normal(p_alpha['a'][i],a_sigma)
                    g2=np.random.normal(p_prime['a'][i],a_sigma)
                    elem=np.exp(L1+np.log(P1)-L2-np.log(P2))*g1/g2
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
                    #new_element=(np.random.normal(loc=p_beta['a'][i],scale=a_sigma))%(2*np.pi) #draw random sample from proposal distribution
                    new_element=np.random.normal(loc=p_beta['a'][i],scale=a_sigma) #draw random sample from proposal distribution
                    p_prime=p_beta.copy()
                    p_prime['a'][i]=new_element
                    #Likelihood
                    L1=Likelihood(**p_beta)
                    L2=Likelihood(**p_prime)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    P1=uniform(p_beta['a'][i])
                    P2=uniform(p_prime['a'][i])                    
                    #Candidates
                    g1= np.random.normal(p_beta['a'][i],a_sigma)
                    g2=np.random.normal(p_prime['a'][i],a_sigma)
                    elem=np.exp(L1+np.log(P1)-L2-np.log(P2))*g1/g2
                    T=min(1,elem)
                    move_prob=random_coin(T)
                    if move_prob:
                        p_beta=p_prime
            if k == 'b': #If it is b's
                for i in range(len(v)):
                    new_element=np.random.normal(loc=p_beta['b'][i],scale=b_sigma) #draw random sample from proposal distribution
                    p_prime=p_beta.copy()
                    p_prime['b'][i]=new_element
                    #Likelihood
                    L1=Likelihood(**p_alpha)
                    L2=Likelihood(**p_prime)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    P1=uniform(p_beta['b'][i])
                    P2=uniform(p_prime['b'][i])
                    #Candidates
                    g1= np.random.normal(p_beta['b'][i],b_sigma)
                    g2=np.random.normal(p_prime['b'][i],b_sigma)
                    elem=np.exp(L1+np.log(P1)-L2-np.log(P2))*g1/g2
                    T=min(1,elem)
                    move_prob=random_coin(T)
                    if move_prob:
                        p_beta=p_prime
    return p_beta

#In this variant the MCMC functionality returns an array due to some strange error otherwise.
def Alg4(Niters,Markov=False,ReturnAll=False,**p):
    
    #This Algorithm is the Metropolis-Hastings within Gibbs sampling algorithm that is 
    #described on the middle of page 94 in Alex Neville's thesis.
    
    if Markov: #i.e. if Markov==True
        chain=Alg4(Niters, Markov=False, ReturnAll=True,**p)
        #print(MCMC) #This is what generates the (wrong) repeated result
        chain=np.stack(chain)
        return chain
    else: #If Markov==False
        MCMC=[]
        #print(MCMC)
        #print("start of Alg 4 p is: {}".format(p))
        for n in range(Niters):
            for k,v in p.items():
                if k == 'eta': #If it is eta's
                    for i in range(len(v)):
                        new_element=np.random.normal(loc=p['eta'][i],scale=eta_sigma) #draw random sample from proposal distribution
                        p_prime=p.copy()
                        p_prime['eta'][i]=new_element
                        #Likelihood
                        L1=Likelihood(**p)
                        L2=Likelihood(**p_prime)
                        #Priors
                        #eta: mu=0.5,sigma=0.05
                        #a: uniform so N/A
                        #b: mu=0.7,sigma=0.07
                        P1=uniform(p['eta'][i])
                        P2=uniform(p_prime['eta'][i])
                        #Candidates
                        g1= np.random.normal(p['eta'][i],eta_sigma)
                        g2=np.random.normal(p_prime['eta'][i],eta_sigma)
                        elem=np.exp(L1+np.log(P1)-L2-np.log(P2))*g1/g2
                        T=min(1,elem)
                        move_prob=random_coin(T)
                        if move_prob:
                            p=p_prime
                if k == 'a': #If it is a's
                    for i in range(len(v)):
                        new_element=np.random.normal(loc=p['a'][i],scale=a_sigma) #draw random sample from proposal distribution
                        p_prime=p.copy()
                        p_prime['a'][i]=new_element
                        #Likelihood
                        L1=Likelihood(**p)
                        L2=Likelihood(**p_prime)
                        #Priors
                        #eta: mu=0.5,sigma=0.05
                        #a: uniform so N/A
                        #b: mu=0.7,sigma=0.07
                        P1=uniform(p['a'][i])
                        P2=uniform(p_prime['a'][i])
                        #Candidates
                        g1= np.random.normal(p['a'][i],a_sigma)
                        g2=np.random.normal(p_prime['a'][i],a_sigma)
                        elem=np.exp(L1+np.log(P1)-L2-np.log(P2))*g1/g2
                        T=min(1,elem)
                        move_prob=random_coin(T)
                        if move_prob:
                            p=p_prime
                if k == 'b': #If it is b's
                    for i in range(len(v)):
                        new_element=np.random.normal(loc=p['b'][i],scale=b_sigma) #draw random sample from proposal distribution
                        p_prime=p_alpha.copy()
                        p_prime['b'][i]=new_element
                        #Likelihood
                        L1=Likelihood(**p)
                        L2=Likelihood(**p_prime)
                        #Priors
                        #eta: mu=0.5,sigma=0.05
                        #a: uniform so N/A
                        #b: mu=0.7,sigma=0.07
                        P1=uniform(p['b'][i])
                        P2=uniform(p_prime['b'][i])                        
                        #Candidates
                        g1= np.random.normal(p['b'][i],b_sigma)
                        g2=np.random.normal(p_prime['b'][i],b_sigma)
                        elem=np.exp(L1+np.log(P1)-L2-np.log(P2))*g1/g2
                        T=min(1,elem)
                        move_prob=random_coin(T)
                        if move_prob:
                            p=p_prime
            
            #print("final p is {}".format(p))
            _=removekey(p,'V')

            etaarray=np.array(_['eta'])
            aarray=np.array(_['a'])
            barray=np.array(_['b'])

            array=np.concatenate((etaarray,aarray,barray))
            #array=np.array(array)
            MCMC.append(array)
            #print(p)
            #print(MCMC)
        #Next turn the reduced dictionary into a simple array for now

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
        p_prime=p_alpha.copy() #Need to use list, lesson learned after 4 years of coding
        new=[np.random.uniform(low=-np.pi,high=np.pi) for i in range(len(p_alpha['a']))]
        p_prime['a']=new
        #Likelihood
        L1=Likelihood(**p_alpha)
        L2=Likelihood(**p_prime)
        #Priors
        #eta: mu=0.5,sigma=0.05
        #a: uniform so N/A
        #b: mu=0.7,sigma=0.07
        P1=1
        for i in range(len(p_alpha['a'])):
            P1*=uniform(p_alpha['a'][i])
        P2=1
        for i in range(len(p_prime['a'])):
            P2*=uniform(p_prime['a'][i])

        elem=(L1+np.log(P1))-(L2+np.log(P2))
        elem=np.exp(elem)

        T=min(1,elem)
        
        #If negative then logT will be the value,otherwise logT will be zero so T will be 1
        #could run a modification so it is accept/reject rather than the most formal version of transition probabilities
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
                    P1=uniform(p_alpha['a'][i])
                    P2=uniform(p_prime['a'][i])
                    elem=(L1+np.log(P1))-(L2+np.log(P2))
                    elem=np.exp(elem)
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
        test=p_alpha.copy()
        for i in range(len(test['a'])):
            test['a'][i]+=q[i]
        #Likelihood
        L1=Likelihood(**test)
        L2=Likelihood(**p_alpha)
        #Priors
        #eta: mu=0.5,sigma=0.05
        #a: uniform so N/A
        #b: mu=0.7,sigma=0.07
        P1=1
        for i in range(len(test['a'])):
            P1*=uniform(test['a'][i])
        P2=1
        for i in range(len(p_alpha['a'])):
            P2*=uniform(p_alpha['a'][i])

        #Am reworking the inequality since np.exp(likelihood) takes it to zero so i am logging the priors instead since it doesn't alter the inequality
        if (L1+np.log(P1))>(L2+np.log(P2)):
        #if (np.exp(L1)*P1)>(np.exp(L2)*P2):
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
    p_alpha=Alg4_alpha(I[3],**p_alpha)
    #print(p_alpha)
    #print("###step 2.2iii done###")
    #step 2.2iv (and 2.2v)
    p_alpha=Alg7(I[4],**p_alpha)
    #print(p_alpha)
    #print("###step 2.2iv done###")

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

#step 2.5
p_zero=p_beta.copy()
for i in range(len(p_zero['eta'])):
    p_zero['eta'][i]=0.5
#print("p_zero is: {}".format(p_zero))

#step 2.6
#p_zero=Alg4(I[6], Markov=False, ReturnAll=False, **p_zero)
p_zero=Alg4(I[6], Markov=False, **p_zero)
#print(p_zero)
#print("###step 2.6 done###")
#step 2.7
p_conv=p_zero.copy()
#print("p_conv is: {}".format(p_conv))

###Main Markov Chain Generation###

#Step 3
chain=Alg4(I[7], Markov=True,ReturnAll=True,**p_conv)

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
#i reckon for now i shall have to use the totalorder and count which occurrence 'BS' or 'PS' it is and then use that to format a string.

names=[]
for _ in range(len(p_conv['eta'])):
    names.append("eta{}".format(_+1))
for _ in range(len(p_conv['a'])):
    names.append("a{}".format(_+1))
for _ in range(len(p_conv['b'])):
    names.append("b{}".format(_+1))
#print(chain) #chain seems to be just a repetition of the same default dicts

def Plot(chain): #Chain should contain all necessary markov chain data
    """
    Custom plot function to generate the standard traceplot format where there is a smoothed
    histogram for each parameter in the left hand column and the markov chain plot in the right hand column.
    e.g. https://python.arviz.org/en/stable/examples/plot_trace.html
    """
    #nitems=len(p_alpha['eta'])+len(p_alpha['a'])+len(p_alpha['b'])
    #fig,axs=plt.subplots(nitems,2,constrained_layout=True) #Can use sharex,sharey for further polish if wanted
    fig,axs=plt.subplots(len(chain[0]),2,constrained_layout=True) #Can use sharex,sharey for further polish if wanted
    #for i in range(nitems):
    for i in range(len(chain[0])):
        #histogram
        #axs[i,0].hist(chain[:,i],bins=30)
        #smoothed kde with scott's rule bandwidth selection (bw selection is important consideration in kde)
        eval_points = np.linspace(np.min(chain[:,i]), np.max(chain[:,i]),len(chain[:,i]))
        kde=gaussian_kde(chain[:,i])
        evaluated=kde.evaluate(eval_points)
        evaluated/=sum(evaluated) #For normalisation
        axs[i,0].plot(eval_points,evaluated)
        #Add axs polish like axes labelling
        axs[i,0].set_ylabel(str(names[i])) #Add label to show which parameter is which
        axs[i,1].plot(chain[:,i])
        axs[i,1].set_xlabel("Markov chain State Number") #Aid understanding of Markov chain plot
    #fig.tight_layout()
    plt.show()

#print(chain)
"""
chain contains an array of default dictionaries so i sgould figure out how to handle this output.
"""

chain=np.array(chain)
#print(chain)
Plot(chain)

for i in range(len(chain[0])): #step 4
    par_array=chain[:,i]
    #Plot markov chain plot
    print(names[i])
    #plt.plot(par_array)
    print("Mean is {}".format(np.mean(par_array)))
    print("Standard deviation is {}".format(np.std(par_array)))