"""
Code to improve MLE method by doing PS characterisation separately via curve fitting, ideally would want bayesian to encapsulate uncertainty but MLE is MLE :(.

PS characterisation will be getting data to fit:
Simple: P_opt = A cos(theta(V)) + B
Expanded: P_opt = A cos(C P(V) - theta_0) + B

P(V)=aV+bV**2+cV**3

fitting A,B,C,theta_0. P_opt is output power, P is power dissipated by heater, governed by above eq.
Output port power mustn't be interfered, this process is for internal phase shifters, external phase shifters must be done via construction of a meta MZI
"""

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

RANDOM_SEED=0
rng = np.random.default_rng(RANDOM_SEED)

N=1000

a=2E-4
b=2E-4
c=2E-4

A=0.5
B=0.5
#C=200
C=-200 #200 and -200 give same result = locally identifiable
theta_0=0

V=np.random.uniform(low=0,high=5,size=N)
Pow_true=a*V+b*(V**2)+c*(V**3)
#Add noise
Pow_noise = Pow_true+rng.normal(scale=0.01, size=N)

#plt.plot(V,Pow_true,linestyle='None',marker=".",markersize=10.0)
#plt.show()
theta=C*Pow_true - theta_0
#Popt_true=A*np.cos(C*Pow_true - theta_0)+B
Popt_true=A*np.cos(theta)+B
#plt.plot(theta,Popt_true,linestyle='None',marker=".",markersize=10.0)
#plt.plot(Pow_true,Popt_true,linestyle='None',marker=".",markersize=10.0)
#plt.show()

def func(x,A,B,C,theta_0):
    return A*np.cos(C*x - theta_0)+B

popt, pcov=curve_fit(func,Pow_true,Popt_true)
plt.plot(Pow_true, func(Pow_true, *popt), linestyle='None',marker=".",markersize=10.0)
print(popt)
"""

#Import modules
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize
import scipy.stats
import multiprocessing
import collections
from collections import defaultdict
from matplotlib import cbook, cm
from matplotlib.colors import LightSource
from scipy.optimize import curve_fit

#Priming
cpucount=multiprocessing.cpu_count()
RANDOM_SEED = 0
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")


#Define initialisation parameters
m=2 #Number of circuit modes ('rows' in the circuit)
Vmax=10 #Max voltage setting on phase shifters
N=100 #Top of page 108 ->N=number of experiments

#Supporting functions (keeping simple with dim=2 for now)
 
def construct_PS(phi): #Phase shifter component unitary (only valid for 2 modes)
    mat=np.array(([np.exp(1j*phi),0],[0,1]))
    return mat

def construct_BS(eta): #Beamsplitter component unitary
    mat=np.array(([np.sqrt(eta), 1j*np.sqrt(1-eta)],[1j*np.sqrt(1-eta),np.sqrt(eta)]))
    return mat
    
def constructU(eta1,eta2,phi):
    mat=construct_BS(eta2)@construct_PS(phi)@construct_BS(eta1)
    #mat=construct_BS(eta2)@construct_PS(phi1)@construct_BS(eta1)
    return mat

#print(construct_BS(0.5)@construct_BS(0.3))
#print(construct_BS(0.3)@construct_BS(0.5))

eta1=0.4
eta2=0.61
N=1000

a=2E-4
b=2E-4
c=2E-4

A=0.5
B=0.5
#C=200
C=-200 #200 and -200 give same result = locally identifiable
theta_0=0


#n_phaseshifters=len(a) if isinstance(a, collections.abc.Iterable) else 1
n_phaseshifters=1

#Since each phase shifter has its own set of voltages applied to each other the block of code below makes one array for each phase shifter.
V=[]
for _ in range(n_phaseshifters):
    Velem=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5 for phase shifter
    #print(Velem)
    #Velem=Velem+rng.normal(scale=0.02, size=N) #Adding gaussian noise
    V.append(list(Velem))

V=np.array(V)


Pow_true=a*V+b*(V**2)+c*(V**3)
#Add noise
Pow_noise=Pow_true
#Pow_noise = Pow_true+rng.normal(scale=0.01, size=N)


phi=C*Pow_true - theta_0 #array
#Popt_true=A*np.cos(C*Pow_true - theta_0)+B
Popt_true=A*np.cos(phi)+B #array
#plt.plot(theta,Popt_true,linestyle='None',marker=".",markersize=10.0)
#plt.plot(Pow_true,Popt_true,linestyle='None',marker=".",markersize=10.0)
#plt.show()

def func(x,A,B,C,theta_0):
    return A*np.cos(C*x - theta_0)+B

#Provide good initial guess or bump up allowed iterations
#popt, pcov=curve_fit(func,Pow_noise[0],Popt_true[0])
#plt.plot(Pow_true[0],Popt_true[0],linestyle='None',marker=".",markersize=10.0)
popt, pcov=curve_fit(f=func,xdata=Pow_true[0],ydata=Popt_true[0],p0=[0.45,0.45,190,0])

print(popt) #output params, pcov conveys covariance which could be useful for keeping bayesianish

#print(V)
#Data Generation

#should generalise datagen to track input state but will stick with fixed input for now
#Remember top ket extracts leftmost column, then top bra extracts top element

inputmode=0
#print(V[0])

P_true=np.empty((N,m))
for i in range(N):
    #print(phi) #phi already found earlier in PS characterisation stage
    U_true=constructU(eta1,eta2,phi[0][i])
    #print(U_true)
    P_click1_true=abs(U_true[0][0])**2 #Probability of click in top
    P_click1_true=P_click1_true
    P_click2_true=abs(U_true[1][0])**2 #Probability of click in bottom
    P_click2_true=P_click2_true
              
    P=[P_click1_true,P_click2_true] #power just be getting clicks out multinomially anyway so no point doing multinomial and just indexing out Unitary like expected
    P_true[i]=P



L_dev=4E-9 #delta L, chosen so it coincides with Alex's sd to be ~np.pi/200
wv=1550E-9
a_dev=(2*np.pi*L_dev)/wv

#Now I have an array of P_true so now I just need to define objective func (residuals of output probs from parameters and ones produced by data) scipy.optimise.minimize

def costfunc(x):
    #print(x)
    eta1=x[0]
    eta2=x[1]

    P_test=np.empty((N,m))
    for i in range(N):
        #phis=[]
        #for j in range(len(expanded_dict['PS'])):
        #    phis.append(a+b*np.square(V[]))
        phi_est=popt[2]*Pow_noise - popt[3]
        #U_true=gen_constructU(eta,phis, m, circuit_ordering)        
        U=constructU(eta1,eta2,phi_est[0][i])
        P_click1=abs(U[0][0])**2 #Probability of click in top
        P_click2=abs(U[1][0])**2 #Probability of click in bottom
        P=[P_click1,P_click2]
        P_test[i]=P
    #print(P_test)
    #print(P_true)
    res=0
    for _ in range(N):
        #res+=(1/N)*(P[_]-P_test[_])**2 #want norm of residuals as my obj func to minimise
        #print(res)
        res+=scipy.linalg.norm(P_true[_]-P_test[_],ord=2)
    #print(res)
    return res

#result=scipy.optimize.minimize(fun=costfunc,x0=np.array([0.5,0.5,0,0,0.7,0.7]),method='L-BFGS-B',bounds=[(0,1),(0,1),(-np.pi,np.pi),(-np.pi,np.pi),(-np.inf,np.inf),(-np.inf,np.inf)])
#scipy.optimize.minimize is local optimisation, not global.
#result=scipy.optimize.minimize(fun=costfunc,x0=np.array([0.3,0.52]),method='L-BFGS-B',bounds=[(0,1),(0,1)])
minimizer_kwargs = { "method": "L-BFGS-B","bounds":[(0,1),(0,1)] }
result=scipy.optimize.basinhopping(func=costfunc,x0=np.array([0.9,0.1]),minimizer_kwargs=minimizer_kwargs)
print(result)

"""
Even small noise on power butchers global optimisation to wrong solution, plus can settle on 
{eta1,eta2} being {0.7,0.02} or {0.02,0.66} so some symmetry there as well when you take phase shifters out of the picture. And global optimisation takes longer than local.
"""