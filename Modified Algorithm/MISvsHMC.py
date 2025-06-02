"""
Code to compare MIS and HMC for boson sampling, intent is to figure out if HMC's convergence is good enough to beat MIS for boson sampling

Active notes

- HMC is a continuous algorithm and boson sampling distribution is quite discrete
"""

#Import modules
import numpy as np
from scipy.stats import unitary_group
import random

#Define number of input photons
m=20 #total of modes
n=10 #number of photons


#Initialisation state 20 mode circuit
#|111...000...0>
init=np.concatenate((np.ones((n),dtype=int),np.zeros((m-n),dtype=int)),axis=0) #Looks right, is a column vector
#print(init)

#Define Haar random unitary construction function
U=unitary_group.rvs(m)
#print(U) #Checks out

#Define permanent calculation algorithm
#Alg copied from thewalrus

def perm_ryser(M):  # pragma: no cover
    """
    Returns the permanent of a matrix using the Ryser formula in Gray ordering.

    The code is an re-implementation from a Python 2 code found in
    `Permanent code golf
    <https://codegolf.stackexchange.com/questions/97060/calculate-the-permanent-as-quickly-as-possible>`_
    using Numba.

    Args:
        M (array) : a square array.

    Returns:
        float or complex: the permanent of matrix ``M``
    """
    n = len(M)
    if n == 0:
        return M.dtype.type(1.0)
    # row_comb keeps the sum of previous subsets.
    # Every iteration, it removes a term and/or adds a new term
    # to give the term to add for the next subset
    row_comb = np.zeros((n), dtype=M.dtype)
    total = 0
    old_grey = 0
    sign = +1
    binary_power_dict = [2**i for i in range(n)]
    num_loops = 2**n
    for k in range(0, num_loops):
        bin_index = (k + 1) % num_loops
        reduced = np.prod(row_comb)
        total += sign * reduced
        new_grey = bin_index ^ (bin_index // 2)
        grey_diff = old_grey ^ new_grey
        grey_diff_index = binary_power_dict.index(grey_diff)
        new_vector = M[grey_diff_index]
        direction = (old_grey > new_grey) - (old_grey < new_grey)
        for i in range(n):
            row_comb[i] += new_vector[i] * direction
        sign = -sign
        old_grey = new_grey
    return total
#print(perm_ryser(U)) #looks OK, is enclosed in brackets but can't index tuple to return just number?

#Define proposal distribution
def distinguishableprop(M):
    A=M.copy()
    for i in range(len(M)):
        for j in range(len(M)):
            A[i][j]=abs(M[i][j])**2
    
    return A

#print(distinguishableprop(U)) #checks out

#This preserves number, still need to see if i can cleverly limit the draw to be heralding compatible, would have to modify the key function if so
def proposal_draw_orig(current):
    new=sorted(current.ravel(), key=lambda k: random.random())
    return new

#Define heralding requirement (specify as int due to )
heralding=np.ones((2),dtype=int)

#This proposal draw automatically satisfies heralding conditions
def proposal_draw_new(current,heralding):
    remaining=current.ravel().copy()
    indices = [i for i, x in enumerate(current.ravel()) if x == 1]
    for _ in range(len(heralding.ravel())):
        remaining=np.delete(remaining.ravel(),indices[_])
    
    shuffled=sorted(remaining.ravel(), key=lambda k: random.random())
    new=np.concatenate((heralding.ravel(),shuffled))
    return new

def proposal_draw_general(current,heralding):
    remaining=current.ravel().copy()
    indices_0 = [i for i, x in enumerate(current.ravel()) if x == 0]
    indices_1 = [i for i, x in enumerate(current.ravel()) if x == 1]

    for _ in range(len(heralding.ravel())):
        if heralding.ravel()[_]==0:
            remaining=np.delete(remaining.ravel(),indices_0[_])
        if heralding.ravel()[_]==1:
            remaining=np.delete(remaining.ravel(),indices_1[_])
    
    shuffled=sorted(remaining.ravel(), key=lambda k: random.random())
    new=np.concatenate((heralding.ravel(),shuffled))
    return new    
#print(proposal_draw_new(initial,heralding))
#proposal_draw_general(init,heralding)

#Initial tuple starting point for sampling
current=init.copy()
results=[current]

#Get initial permanent result for output
indices_initial = [i for i, x in enumerate(current.ravel()) if x == 1]
U_init = U[np.ix_([i for i in range(n)],indices_initial)]
#print(U)
#print()
#print(U_init)
res=-np.log(abs(perm_ryser(U_init))**2)
perms=[res]

N=10000 #Number of iterations

def MIS(N,initial_State):
    results_MIS=results.copy()
    perms_MIS=perms.copy()
    current=init.copy()
    for _ in range(N):
        #print(_)
        #Obtain new proposed state
        #new=proposal_draw_new(current,heralding)
        new=proposal_draw_orig(current)

        #Obtain indices of 1s in state arrays
        indices_current = [i for i, x in enumerate(current.ravel()) if x == 1]
        indices_new = [i for i, x in enumerate(new.ravel()) if x == 1]

        #Isolate submatrix
        U_curr = U[np.ix_([i for i in range(n)],indices_current)]
        U_new = distinguishableprop(U)[np.ix_([i for i in range(n)],indices_new)]

        #Transition probability evaluation + appropriate log-Per calc
        true_curr=abs(perm_ryser(U_curr))**2
        prop_curr=perm_ryser(distinguishableprop(U_curr))
        true_new=abs(perm_ryser(U_new))**2
        prop_new=perm_ryser(distinguishableprop(U_new))
        frac=(true_new*prop_curr)/(true_curr*prop_new)
        Tprob=min(1,frac)

        #Conduct result of T evaluation (biased coin flip)
        u=np.random.uniform()
        if u<=Tprob:
            results_MIS.append(new)
            perms_MIS.append(-np.log(true_new))
            current=new.copy()
        else:
            results_MIS.append(current)
            perms_MIS.append(-np.log(true_curr))
    return results_MIS,perms_MIS

results_MIS,perms_MIS=MIS(N,init)

#######################################
#HMC
import scipy.stats as st

def neg_log_prob(x,mu,sigma):
    return -1*np.log(normal(x=x,mu=mu,sigma=sigma))

def HMC(mu=0.0,sigma=1.0,path_len=1,step_size=0.25,initial_position=0.0,epochs=1_000):
    # setup
    steps = int(path_len/step_size) # path_len and step_size are tricky parameters to tune...
    samples = [initial_position]
    momentum_dist = st.norm(0, 1) 
    # generate samples
    for e in range(epochs):
        q0 = np.copy(samples[-1])
        q1 = np.copy(q0)
        p0 = momentum_dist.rvs()        
        p1 = np.copy(p0) 
        dVdQ = -1*(q0-mu)/(sigma**2) # gradient of PDF wrt position (q0) aka potential energy wrt position

        # leapfrog integration begin
        for s in range(steps): 
            p1 += step_size*dVdQ/2 # as potential energy increases, kinetic energy decreases, half-step
            q1 += step_size*p1 # position increases as function of momentum 
            p1 += step_size*dVdQ/2 # second half-step "leapfrog" update to momentum    
        # leapfrog integration end        
        p1 = -1*p1 #flip momentum for reversibility     

        
        #metropolis acceptance
        q0_nlp = neg_log_prob(x=q0,mu=mu,sigma=sigma)
        q1_nlp = neg_log_prob(x=q1,mu=mu,sigma=sigma)        

        p0_nlp = neg_log_prob(x=p0,mu=0,sigma=1)
        p1_nlp = neg_log_prob(x=p1,mu=0,sigma=1)
        
        # Account for negatives AND log(probabiltiies)...
        target = q0_nlp - q1_nlp # P(q1)/P(q0)
        adjustment = p1_nlp - p0_nlp # P(p0)/P(p1)
        acceptance = target + adjustment # [P(q1)*P(p0)]/[P(q0)*P(p1)] 
        
        event = np.log(random.uniform(0,1))
        if event <= acceptance:
            samples.append(q1)
        else:
            samples.append(q0)
    
    return samples

###############################################

#print(results)
def thin(array,interval):
    return array[interval-1::interval]

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

eval_points = np.linspace(np.min(perms), np.max(perms),len(perms))
kde=gaussian_kde(perms)
eval_points = np.linspace(np.min(perms), np.max(perms))
evaluated=kde.evaluate(eval_points)
evaluated/=sum(evaluated) #For normalisation
print(sum(evaluated))
plt.plot(eval_points,evaluated)
#plt.plot(eval_points,kde.pdf(eval_points)/sum(kde.pdf(eval_points)))
#binwidth=0.5
#plt.hist(perms,bins=int(((np.max(perms)-np.min(perms))/binwidth)), density=True,histtype='bar', ec='black')
plt.xlim([np.mean(perms)-np.std(perms),np.mean(perms)-np.std(perms)])
plt.xlabel("-log(|Per(A_S)|^2)")
plt.ylabel("Probability")
plt.show()