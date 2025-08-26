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

#a = 2E-4
#b = 2E-4
#c = 2E-4

A_res=np.zeros((N,)) 
B_res=np.zeros((N,))
C_res=np.zeros((N,))
theta0_res=np.zeros((N,))
a_res=np.zeros((N,)) 
b_res=np.zeros((N,))
c_res=np.zeros((N,))

a_true=np.zeros((N,))
a_true+=1+abs(rng.normal(scale=1E-3, size=(N,)))
b_true=np.zeros((N,))
b_true+=.01+abs(rng.normal(scale=1E-3, size=(N,)))
c_true=np.zeros((N,))
c_true+=.001+abs(rng.normal(scale=1E-3, size=(N,)))

A_true=np.zeros((N,))
A_true+=1+rng.normal(scale=.1, size=(N,))
#print(A_true)
B_true=np.zeros((N,))
B_true+=rng.normal(scale=.1, size=(N,))
C_true=np.zeros((N,))
C_true+=1+rng.normal(scale=1, size=(N,))
#print(C_true)
theta0_true=np.zeros((N,))
theta0_true+=rng.normal(scale=.1, size=(N,))

n_phaseshifters = 1

V = np.random.uniform(low=0, high=Vmax, size=(n_phaseshifters, N))  # without noise, value we believe we are setting
#print(V[0])
#no noise
V_noisy=V
#noise
#V_noisy = V + rng.normal(scale=0.05, size=(n_phaseshifters, N))  # extra noise on voltage, value actually set

def func1(x, A, B, C, theta_0,a,b,c):
    return A * np.cos(C * (a * x + b * (x ** 2) + c * (x ** 3)) - theta_0) + B

def func2(x, A, B, C, theta_0):
    return A * np.cos(C * (a * x + b * (x ** 2) + c * (x ** 3)) - theta_0) + B

def cost1(x): #MSE
    A,B,C,theta_0,a,b,c=x
    return np.mean((func1(V_noisy[0],A,B,C, theta_0,a,b,c)-P_opt)**2)

def cost2(x): #MSE
    A,B,C,theta_0=x
    return np.mean((func2(V_noisy[0],A,B,C, theta_0)-P_opt)**2)

Popt_true=np.zeros((N,N))
Popt_res=np.zeros((N,N))


#including a,b,c
for _ in range(N):
    P_elect = a_true[_] * V_noisy + b_true[_] * (V_noisy ** 2) + c_true[_] * (V_noisy ** 3)  # electrical power
    phi = C_true[_] * P_elect - theta0_true[_]  # phase
    P_opt = A_true[_] * np.cos(phi)**2 + B_true[_]  # optical power
    Popt_true[_]=P_opt
    p0 = [1, 0, 0,0,1,0,0]  # initial guess
    #res = scipy.optimize.minimize(cost, p0)
    minimizer_kwargs = { "method": "L-BFGS-B"}
    res=scipy.optimize.basinhopping(func=cost1,x0=p0,minimizer_kwargs=minimizer_kwargs)
    #print(res.x)  # see if minimization succeeded.
    A_res[_]=res.x[0]
    B_res[_]=res.x[1]
    C_res[_]=res.x[2]
    theta0_res[_]=res.x[3]
    a_res[_]=res.x[4]
    b_res[_]=res.x[5]
    c_res[_]=res.x[6]

    P_temp = a_res[_] * V_noisy + b_res[_] * (V_noisy ** 2) + c_res[_] * (V_noisy ** 3)  # electrical power
    phi_temp = C_res[_] * P_temp - theta0_res[_]  # phase
    Popt_res[_] = A_res[_] * np.cos(phi_temp)**2 + B_res[_]


#euclidean distance again?
#eucliddist=np.zeros((N,))
a=np.stack([A_true-A_res,B_true-B_res,C_true-C_res,theta0_true-theta0_res,a_true-a_res,b_true-b_res,c_true-c_res])
eucliddist=np.linalg.norm(a,axis=0)
#print(eucliddist)
#print(np.sqrt(np.sum(np.square(Popt_true[0]-Popt_res[0]))))
Pdiff=np.linalg.norm(Popt_true-Popt_res,axis=1) #two arrays, so take MSE
#print(Pdiff)


fig,ax=plt.subplots()
ax.scatter(eucliddist,Pdiff)

plt.show()

"""
#excluding a,b,c
for _ in range(N):
    P_elect = a_true[_] * V_noisy + b_true[_] * (V_noisy ** 2) + c_true[_] * (V_noisy ** 3)  # electrical power
    phi = C_true[_] * P_elect - theta0_true[_]  # phase
    P_opt = A_true[_] * np.cos(phi)**2 + B_true[_]  # optical power
    #print(P_opt)
    Popt_true[_]=P_opt
    a=a_true[_]
    b=b_true[_]
    c=c_true[_]
    p0 = [1, 0, 0,0]  # initial guess
    #res = scipy.optimize.minimize(cost, p0)
    minimizer_kwargs = { "method": "L-BFGS-B"}
    res=scipy.optimize.basinhopping(func=cost2,x0=p0,minimizer_kwargs=minimizer_kwargs)
    #print(res)  # see if minimization succeeded.
    #print([A_true[_],B_true[_],C_true[_],theta0_true[_]])
    #print([res.x[0],res.x[1],res.x[2],res.x[3]])
    A_res[_]=res.x[0]
    B_res[_]=res.x[1]
    C_res[_]=res.x[2]
    theta0_res[_]=res.x[3]

    P_temp = a_true[_] * V_noisy + b_true[_] * (V_noisy ** 2) + c_true[_] * (V_noisy ** 3)  # electrical power
    phi_temp = C_res[_] * P_temp - theta0_res[_]  # phase
    Popt_result = A_res[_] * np.cos(phi_temp)**2 + B_res[_]
    #print(Popt_result)
    Popt_res[_]=Popt_result


#euclidean distance again?
#eucliddist=np.zeros((N,))
a=np.stack([A_true-A_res,B_true-B_res,C_true-C_res,theta0_true-theta0_res])
eucliddist=np.linalg.norm(a,axis=0)
#print(eucliddist)
#print(np.sqrt(np.sum(np.square(Popt_true[0]-Popt_res[0]))))
Pdiff=np.linalg.norm(Popt_true-Popt_res,axis=1) #two arrays, so take MSE
#print(Pdiff)


fig,ax=plt.subplots()
ax.scatter(eucliddist,Pdiff)

plt.show()
"""


"""
comparing results like below doesn't work too good since multiple optimal solutions, 
so again want to evaluate performance through how well it fits the curve, AKA difference 
between P_opt and that which is the cost function. So call via res.fun
"""
#print(A_true-A_res)
#print(B_true-B_res)
#print(C_true-C_res)
#print(theta0_true-theta0_res)


"""
Basin hopping is a global method but lowest optimisation result from print(res) has 
Hessian Inverse 'hess_inv', need to use todense() to make a matrix array, then you 
can obtain the standard error of the estimated parameters by taking the sqrt of the 
diagonal of the Hessian inverse 
"""
#print(np.sqrt(np.diag(res.lowest_optimization_result.hess_inv.todense())))