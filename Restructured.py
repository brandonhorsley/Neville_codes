"""
Restructured code to make the code more robust. This will include altering my code to be more like 
Emilien's in too main ways. One is basically put the probabilities in log form and by extension the 
acceptance rules. Secondly is putting it into posterior form rather than P1*L1*g1 for example.

Need to make sure my indentations are proper in the subalgorithm parts

Definitely doesn't work
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt

#####################Generating Data#######################

RANDOM_SEED = 8927 #Seed for numpy random
rng = np.random.default_rng(RANDOM_SEED)

#########################Supporting functions

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

def ConstructU(eta1,eta2,eta3,phi1,phi2):
    U=construct_BS(eta3)@construct_PS(phi2)@construct_BS(eta2)@construct_PS(phi1)@construct_BS(eta1)
    return U

#####################Data generation
Vmax=5
N=100 #Top of page 108 ->N=number of experiments
M=2 #Number of modes
V1=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5 for phase shifter 1
#V1=V1+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
V2=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5 for phase shifter 2
#V2=V2+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise

a1_true=0
a2_true=0

b1_true=0.788
b2_true=0.711

eta1_true=0.447
eta2_true=0.548
eta3_true=0.479

def DataGen(InputNumber, V1, V2,poissonian=False): #InputNumber=# of input photons= should average to about 1000
    data=np.empty((N,M))
    C=np.empty(N)

    for i in range(N):
        #Input into mth mode of beamsplitter
        phi1_true=a1_true+b1_true*V1[i]**2 #phi=a+bV**2
        phi2_true=a2_true+b2_true*V2[i]**2 #phi=a+bV**2
        U_true=ConstructU(eta1_true,eta2_true,eta3_true,phi1_true,phi2_true) #Generate double MZI Unitary
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

#data,C=DataGen(InputNumber=1000,Voltages=V,poissonian=False)
data,C=DataGen(InputNumber=1000,V1=V1,V2=V2,poissonian=False)
#print(data) #Correct
#print(C)

def Likelihood(p,V1,V2):
    eta1=p[0]
    eta2=p[1]
    eta3=p[2]
    a1=p[3]
    a2=p[4]
    b1=p[5]
    b2=p[6]
    #To be called after data generation
    P=np.empty((N,M))
    prob=np.empty(N)

    for i in range(len(V1)): #len(V1) should equal len(V2)
        phi1=a1+b1*V1[i]**2 #phi=a+bV**2
        phi2=a2+b2*V2[i]**2 #phi=a+bV**2
        U=ConstructU(eta1,eta2,eta3,phi1,phi2) #Generate double MZI Unitary
        P_click1=np.abs(top_bra@U@top_ket)**2 #Probability of click in top
        P_click1=P_click1[0][0]
        P_click2=np.abs(bottom_bra@U@top_ket)**2 #Probability of click in bottom
        P_click2=P_click2[0][0]
        P[i]=[P_click1,P_click2]
        #n=C,p=P,x=array of clicks
        prob[i]=scipy.stats.multinomial.logpmf(x=data[i],n=C[i],p=P[i])
        #if np.isinf(prob[i]):
        #    prob[i]=0 #To bypass -inf ruining likelihood calculations.
    #print(prob)
    logsum=np.sum(prob)
    return logsum

#eq 4.11: g_i(p',p)=Normal(p_i,sigma_i) #proposal distribution
#sigma_i=pi/200 for a, b_est for b, 0.005 for eta

b_est=0.7
eta_sigma=0.005
a_sigma=np.pi/200
b_sigma=b_est #Based around true values from Neville_thesis_8.py

#I=[2,500,50,50,500,100,100,100000] #Determines iteration number for each algorithm call
I=[2,500,50,50,500,100,100,10000] #Smaller MCMC chain for troubleshooting

###Burn in###

p_alpha=[0.5,0.5,0.5,0,0,0.7,0.7] #step 2.1
print("p_alpha initial is {}".format(p_alpha))

def Alg4_alpha(p_alpha, Niters):
    """
    Algorithm variant of Algorithm 4 for estimating the alpha model (which involves 
    only learning the a values).
    """
    for n in range(Niters):
        for i in range(len(p_alpha)):
            if i in [3,4]: #If it is a's
                new_element=np.random.normal(loc=p_alpha[i],scale=a_sigma) #draw random sample from proposal distribution
                p_prime=list(p_alpha) 
                p_prime[i]=new_element
                
                L1=Likelihood(p_alpha,V1,V2)
                L2=Likelihood(p_prime,V1,V2)
                
                P1=scipy.stats.uniform.logpdf(p_alpha[i],loc=-np.pi,scale=2*np.pi)
                P2=scipy.stats.uniform.logpdf(p_prime[i],loc=-np.pi,scale=2*np.pi)
                
                g1=scipy.stats.norm.logpdf(new_element, p_alpha[i], a_sigma)
                g2=scipy.stats.norm.logpdf(new_element, p_prime[i], a_sigma)
                
                post1=L1+P1
                post2=L2+P2
                
                term=(post1+g1)-(post2+g2)
                if np.log(np.random.uniform(0,1)) <= term:
                    p_alpha=p_prime
    return p_alpha

def Alg4_beta(p_beta, Niters):
    """
    Algorithm variant of Algorithm 4 for estimating the beta model (which involves 
    learning the a and b values).
    """
    for n in range(Niters):
        for i in range(len(p_beta)):
            if i in [3,4]: #If it is a's
                new_element=np.random.normal(loc=p_beta[i],scale=a_sigma) #draw random sample from proposal distribution
                p_prime=list(p_beta)
                p_prime[i]=new_element #new proposed state
                
                L1=Likelihood(p_beta,V1,V2)
                L2=Likelihood(p_prime,V1,V2)
                
                P1=scipy.stats.uniform.logpdf(p_beta[i],loc=-np.pi,scale=2*np.pi)
                P2=scipy.stats.uniform.logpdf(p_prime[i],loc=-np.pi,scale=2*np.pi)
                
                g1=scipy.stats.norm.logpdf(new_element, p_beta[i], a_sigma)
                g2=scipy.stats.norm.logpdf(new_element, p_prime[i], a_sigma)
                
                post1=L1+P1
                post2=L2+P2
                
                term=(post1+g1)-(post2+g2)
                if np.log(np.random.uniform(0,1)) <= term:
                    p_beta=p_prime
            if i in [5,6]: #If it is b's
                new_element=np.random.normal(loc=p_beta[i],scale=b_sigma) #draw random sample from proposal distribution
                p_prime=list(p_beta)
                p_prime[i]=new_element #new proposed state
                
                L1=Likelihood(p_beta,V1,V2)
                L2=Likelihood(p_prime,V1,V2)
                
                P1=scipy.stats.norm.logpdf(p_beta[i],loc=0.7,scale=b_sigma)
                P2=scipy.stats.norm.logpdf(p_prime[i],loc=0.7,scale=b_sigma)
                
                g1=scipy.stats.norm.logpdf(new_element, p_beta[i], b_sigma)
                g2=scipy.stats.norm.logpdf(new_element, p_prime[i], b_sigma)
                
                post1=L1+P1
                post2=L2+P2
                
                term=(post1+g1)-(post2+g2)
                if np.log(np.random.uniform(0,1)) <= term:
                    p_beta=p_prime
    return p_beta

def Alg4(p,Niters,Markov=False,ReturnAll=False):
    """
    This Algorithm is the Metropolis-Hastings within Gibbs sampling algorithm that is 
    described on the middle of page 94 in Alex Neville's thesis.
    """
    if Markov: #i.e. if Markov==True
        MCMC=Alg4(p,Niters, Markov=False, ReturnAll=True)
        return MCMC
    else: #If Markov==False
        MCMC=[]
        for n in range(Niters):
            for i in range(len(p)):
                #print(i)
                if i in [0,1,2]: #If it is eta's
                    #print("enter eta")
                    new_element=np.random.normal(loc=p[i],scale=eta_sigma) #draw random sample from proposal distribution
                    p_prime=list(p)
                    p_prime[i]=new_element #new proposed state
                    
                    L1=Likelihood(p,V1,V2)
                    L2=Likelihood(p_prime,V1,V2)
                    
                    P1=scipy.stats.norm.logpdf(p[i],loc=0.7,scale=eta_sigma)
                    P2=scipy.stats.norm.logpdf(p_prime[i],loc=0.7,scale=eta_sigma)
                    
                    g1=scipy.stats.norm.logpdf(new_element, p[i], eta_sigma)
                    g2=scipy.stats.norm.logpdf(new_element, p_prime[i], eta_sigma)
                    
                    post1=L1+P1
                    post2=L2+P2
                    
                    term=(post1+g1)-(post2+g2)
                    if np.log(np.random.uniform(0,1)) <= term:
                        p=p_prime

                if i in [3,4]: #If it is a's
                    #print("enter a")
                    new_element=np.random.normal(loc=p[i],scale=a_sigma) #draw random sample from proposal distribution
                    p_prime=list(p)
                    p_prime[i]=new_element #new proposed state
                    
                    L1=Likelihood(p,V1,V2)
                    L2=Likelihood(p_prime,V1,V2)
                    
                    P1=scipy.stats.uniform.logpdf(p[i],loc=-np.pi,scale=2*np.pi)
                    P2=scipy.stats.uniform.logpdf(p_prime[i],loc=-np.pi,scale=2*np.pi)
                    
                    g1=scipy.stats.norm.logpdf(new_element, p[i], a_sigma)
                    g2=scipy.stats.norm.logpdf(new_element, p_prime[i], a_sigma)
                    
                    post1=L1+P1
                    post2=L2+P2
                    
                    term=(post1+g1)-(post2+g2)
                    if np.log(np.random.uniform(0,1)) <= term:
                        p=p_prime

                if i in [5,6]: #If it is b's
                    #print("enter b")
                    new_element=np.random.normal(loc=p[i],scale=b_sigma) #draw random sample from proposal distribution
                    p_prime=list(p)
                    p_prime[i]=new_element #new proposed state
                    
                    L1=Likelihood(p,V1,V2)
                    L2=Likelihood(p_prime,V1,V2)
                    
                    P1=scipy.stats.norm.logpdf(p[i],loc=0.7,scale=b_sigma)
                    P2=scipy.stats.norm.logpdf(p_prime[i],loc=0.7,scale=b_sigma)
                    
                    g1=scipy.stats.norm.logpdf(new_element, p[i], b_sigma)
                    g2=scipy.stats.norm.logpdf(new_element, p_prime[i], b_sigma)
                    
                    post1=L1+P1
                    post2=L2+P2
                    
                    term=(post1+g1)-(post2+g2)
                    if np.log(np.random.uniform(0,1)) <= term:
                        p=p_prime

            if ReturnAll:
                MCMC.append(p)

        if ReturnAll:
            return MCMC
        else:
            return p

def Alg5(p_alpha,Niters):
    """
    This Algorithm is the Metropolised Independence Sampling (MIS) algorithm that is
    described at the top of page 98 in Alex Neville's thesis.
    """
    for n in range(Niters):
        p_prime=list(p_alpha) #new proposed state
        new=[np.random.uniform(low=-np.pi,high=np.pi) for i in range(2)]
        p_prime[3]=new[0]
        p_prime[4]=new[1]

        L1=Likelihood(p_alpha,V1,V2)
        L2=Likelihood(p_prime,V1,V2)

        P1=scipy.stats.uniform.logpdf(p_alpha[3],loc=-np.pi,scale=2*np.pi)+scipy.stats.uniform.logpdf(p_alpha[4],loc=-np.pi,scale=2*np.pi)
        P2=scipy.stats.uniform.logpdf(p_prime[3],loc=-np.pi,scale=2*np.pi)+scipy.stats.uniform.logpdf(p_prime[4],loc=-np.pi,scale=2*np.pi)

        post1=L1+P1
        post2=L2+P2

        term=post1-post2
        if np.log(np.random.uniform(0,1)) <= term:
            p_alpha=p_prime

    return p_alpha

def Alg6(p_alpha,Niters):
    """
    This Algorithm is the Metropolised Independence Sampling (MIS) within Gibbs algorithm 
    that is described in the middle of page 98 in Alex Neville's thesis.
    """
    for n in range(Niters):
        for i in range(len(p_alpha)):
            if i in [3,4]: #If it is a's
                new_element=np.random.uniform(low=-np.pi,high=np.pi) #draw random sample from proposal distribution
                p_prime=list(p_alpha) 
                p_prime[i]=new_element
                
                L1=Likelihood(p_alpha,V1,V2)
                L2=Likelihood(p_prime,V1,V2)
                
                P1=scipy.stats.uniform.logpdf(p_alpha[i],loc=-np.pi,scale=2*np.pi)
                P2=scipy.stats.uniform.logpdf(p_prime[i],loc=-np.pi,scale=2*np.pi)
                
                post1=L1+P1
                post2=L2+P2
                
                term=post1-post2
                if np.log(np.random.uniform(0,1)) <= term:
                    p_alpha=p_prime

    return p_alpha

def Alg7(p_alpha, Niters):
    """
    This Algorithm is the stochastic π kick search algorithm that is described 
    at the bottom of page 98 in Alex Neville's thesis.
    """
    for n in range(Niters):
        x=np.random.choice(3,2)
        q=[(x[i]-1)*np.pi for i in range(len(x))]

        test=list(p_alpha)
        test[3]+=q[0]
        test[4]+=q[1]

        L1=Likelihood(test,V1,V2)
        L2=Likelihood(p_alpha,V1,V2)

        P1=scipy.stats.uniform.logpdf(test[3],loc=-np.pi,scale=2*np.pi)+scipy.stats.uniform.logpdf(test[4],loc=-np.pi,scale=2*np.pi)
        P2=scipy.stats.uniform.logpdf(p_alpha[3],loc=-np.pi,scale=2*np.pi)+scipy.stats.uniform.logpdf(p_alpha[4],loc=-np.pi,scale=2*np.pi)

        post1=L1+P1
        post2=L2+P2

        if post1>post2:
            p_alpha=test

    return p_alpha

#Main code bulk

for i in range(I[0]): #step 2.2
    #step 2.2i
    p_alpha=Alg5(p_alpha,I[1])
    print(p_alpha)
    print("###step 2.2i done###")
    #step 2.2ii
    p_alpha=Alg6(p_alpha,I[2])
    print(p_alpha)
    print("###step 2.2ii done###")
    #step 2.2iii
    p_alpha=Alg4_alpha(p_alpha, I[3]) #p_alpha is second p_alpha
    print(p_alpha)
    print("###step 2.2iii done###")
    #step 2.2iv (and 2.2v)
    p_alpha=Alg7(p_alpha,I[4])
    print(p_alpha)
    print("###step 2.2iv done###")

p_beta=[0.5,0.5,0.5,p_alpha[3],p_alpha[4],0.7,0.7]

print("p_beta initial is: {}".format(p_beta))
#step 2.4
p_beta=Alg4_beta(p_beta, I[5])
print(p_beta)
print("###step 2.4 done###")

p_zero=[0.5,0.5,0.5,p_beta[3],p_beta[4],p_beta[5],p_beta[6]] #step 2.5
print("p_zero is: {}".format(p_zero))

#step 2.6
p_zero=Alg4(p_zero,I[6], Markov=False)
print(p_zero)
print("###step 2.6 done###")

p_conv=list(p_zero) #step 2.7
print("p_conv is: {}".format(p_conv))


###Main Markov Chain Generation###

#Step 3
chain=Alg4(p_conv,I[7], Markov=True)

###Parameter estimation###

"""
This chain object should contain all the values for each parameter at each 
markov chain state number (i.e. I[7] by 7 matrix).
"""

from scipy.stats import gaussian_kde

names=["eta1","eta2","eta3","a1","a2","b1","b2"]
trues=[eta1_true,eta2_true,eta3_true,a1_true,a2_true,b1_true,b2_true]

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
        axs[i,0].axvline(x=trues[i],c="red")
        #Add axs polish like axes labelling
        axs[i,0].set_ylabel(str(names[i])) #Add label to show which parameter is which
        axs[i,1].plot(chain[:,i])
        axs[i,1].axhline(y=trues[i],c="red")
        axs[i,1].set_xlabel("Markov chain State Number") #Aid understanding of Markov chain plot
    #fig.tight_layout()
    plt.show()

chain=np.array(chain)
Plot(chain)

for i in range(len(p_conv)): #step 4
    par_array=chain[:,i]
    #Plot markov chain plot
    print(names[i])
    #plt.plot(par_array)
    print("Mean is {}".format(np.mean(par_array)))
    print("Standard deviation is {}".format(np.std(par_array)))