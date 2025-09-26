"""
Code file of hardcoding a full characterisation with MLE to compare to Bayes equivalent to see if problem is bayesian or implementation.

- datagen was wrong, need to slice modeU[output,input], and gives a complex number which is why you need complex conjugate and thus curve fit should be Acos^2
-cos squared curve isn't constant due to nonlinearity in IV response leading to changing variable phase, its why in Niko thesis he uses adjusted current ramp, will truncate at 2V as an intermediary solution
"""

#Import modules
import numpy as np
import multiprocessing
import arviz as az
import pymc as pm
import scipy
from scipy.optimize import curve_fit
import time


#Priming
cpucount=multiprocessing.cpu_count()
RANDOM_SEED = 0
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

#Define key parameters and arrays
m=4
N=10000

n_MZI=int(m*(m-1)/2)
V_global=np.zeros((n_MZI,))
V_max=10 #changed from 10V due to nonlinearity problems

A_res=np.zeros((n_MZI,)) 
B_res=np.zeros((n_MZI,))
C_res=np.zeros((n_MZI,))
theta0_res=np.zeros((n_MZI,))
V0_res=np.zeros((n_MZI,))
Vpi_res=np.zeros((n_MZI,))

a_true=np.zeros((n_MZI,))
a_true+=1+abs(rng.normal(scale=1E-3, size=(n_MZI,)))
b_true=np.zeros((n_MZI,))
b_true+=.01+abs(rng.normal(scale=1E-3, size=(n_MZI,)))
c_true=np.zeros((n_MZI,))
c_true+=.001+abs(rng.normal(scale=1E-3, size=(n_MZI,)))

A_true=np.zeros((n_MZI,))
A_true+=1+rng.normal(scale=.1, size=(n_MZI,))
#print(A_true)
B_true=np.zeros((n_MZI,))
B_true+=rng.normal(scale=.1, size=(n_MZI,))
C_true=np.zeros((n_MZI,))
C_true+=1+rng.normal(scale=1, size=(n_MZI,))
#print(C_true)
theta0_true=np.zeros((n_MZI,))
theta0_true+=rng.normal(scale=.1, size=(n_MZI,))

MSE=np.zeros((n_MZI,))


order=[5,4,3,2,1,0] #order of MZIs to multiply to get final unitary
positions=[1,2,0,1,2,0] #top waveguide each MZI is on

V = np.random.uniform(low=0, high=V_max, size=(N,))  # without noise, value we believe we are setting
V_noisy=V
#V_noisy = V + rng.normal(scale=0.05, size=(N,))  # extra noise on voltage, value actually set

#import matplotlib.pyplot as plt

#theta=a_true[0]*V+b_true[0]*V**2+c_true[0]*V**3
#y=A_true[0]*np.cos(C_true[0]*theta-theta0_true[0])**2
#plt.plot(V,y,'s') #is this that is causing the problems, could be a,b,c params leading to too strong nonlinear response?
#plt.show()

def idealMZI(phi): #assume both 50:50 beamsplitters
    #globalfactor=np.exp(1j*(phi/2 + np.pi/2))
    globalfactor=1j
    matrix=np.array([[np.sin(phi/2),np.cos(phi/2)],[np.cos(phi/2),-np.sin(phi/2)]])
    return globalfactor*matrix

def globalU(M,d,position): #embed unitary in terms of full mode unitary, so identity on all other modes
    U=np.eye(d,dtype=np.complex128)
    dim=M.shape
    U[position:position+dim[0],position:position+dim[1]]=M
    return U

def DataGen(MZI_no,input,output):
    #[:,None]
    MatArray=[]
    for i in range(N):
        V_global[MZI_no]=V[i]
        P_elect_true = np.multiply(a_true,V_global) + np.multiply(b_true,V_global**2) + np.multiply(c_true,V_global**3) # electrical power
        #print(P_elect_true)
        phi_true=C_true*P_elect_true-theta0_true

        modeU=np.eye(m,dtype=np.complex128)
        for j in range(len(order)):
            modeU=modeU@globalU(idealMZI(phi_true[order[j]]),m,positions[j])
            #print(modeU)
            #print()
        #print(input)
        #print(output)
        #print(modeU)
        #print(modeU[output,input])
        #print(np.abs(modeU[output,input])**2)
        MatArray.append(np.abs(modeU[output,input])**2)
    return MatArray


#define curve fitting function

def curvefit(MZI_no,InputPort,OutputPort):
    #Produce data
    clickdata=DataGen(MZI_no,InputPort,OutputPort)
    #print(clickdata)
    import matplotlib.pyplot as plt

    #plt.plot(V,clickdata,'s')
    #plt.show()

    P_elect = a_true[0] * V_noisy + b_true[0] * (V_noisy ** 2) + c_true[0] * (V_noisy ** 3)  # electrical power
    phi = C_true[0] * P_elect - theta0_true[0]  # phase
    P_opt = A_true[0] * np.cos(phi)**2 + B_true[0]  # optical power
    Popt_true=P_opt
    
    #do pymc process
    with pm.Model() as model:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        #a = pm.Normal("a", 2E-4,sigma=1)
        #b = pm.Normal("b", 2E-4,sigma=1)
        #c = pm.Normal("c", 2E-4,sigma=1)
        A = pm.Normal("A", 1,sigma=.1)
        B = pm.Normal("B", 0,sigma=.1)
        C = pm.Normal("C", 1,sigma=.1)
        theta_0 = pm.Normal("theta_0",0,sigma=.1)

        # Define likelihood
        #likelihood = pm.Normal("P_opt", mu=A*pm.math.cos(C * (a * V + b * (V ** 2) + c * (V ** 3)) - theta_0) + B, sigma=.1, observed=clickdata)
        #I think model is using wrong voltage array, no accounting for all MZIs?
        likelihood = pm.Normal("P_opt", mu=A*pm.math.cos(C * (a_true[MZI_no] * V + b_true[MZI_no] * (V ** 2) + c_true[MZI_no] * (V ** 3)) - theta_0)**2+B, sigma=.1, observed=clickdata)

        # Inference!
        # draw 3000 posterior samples using NUTS sampling
        idata = pm.sample(3000,cores=1)
    
    az.plot_trace(idata)
    plt.show()
    #return parameters and voltage for complete transmission+complete reflection
    table=az.summary(idata)
    #print(table['mean']['A'])
    #Take max likelihood value for each of the parameters
    #A_max=print(idata.posterior['A'])
    
    #A_max=table['mean']['A']
    #B_max=table['mean']['B']
    #C_max=table['mean']['C']
    #theta_0_max=table['mean']['theta_0']

    #print(A_max)
    #print(B_max)
    #print(C_max)
    #print(theta_0_max)
    
    A_max=scipy.stats.mode(idata.posterior['A'][0],axis=0).mode
    B_max=scipy.stats.mode(idata.posterior['B'][0],axis=0).mode
    C_max=scipy.stats.mode(idata.posterior['C'][0],axis=0).mode    
    theta_0_max=scipy.stats.mode(idata.posterior['theta_0'][0],axis=0).mode
    
    #print(A_max)
    #print(B_max)
    #print(C_max)
    #print(theta_0_max)

    P_temp = a_true[0] * V_noisy + b_true[0] * (V_noisy ** 2) + c_true[0] * (V_noisy ** 3)  # electrical power
    phi_temp = C_max * P_temp - theta_0_max  # phase
    Popt_res = A_max * np.cos(phi_temp)**2 + B_max

    MSE[MZI_no]=np.mean(Popt_res-Popt_true)**2

    #Transmission/reflection means just find point where minimum is or maximum is
    P_ref=(np.pi+theta_0_max)/C_max
    V_ref=np.roots([c_true[MZI_no],b_true[MZI_no],a_true[MZI_no],-P_ref]) #Don't forget the minus sign!
    filtered = [i.real for i in V_ref if i.imag == 0]
    #print(filtered)
    V_ref=filtered[0] #hopefully there will only be one non complex number

    P_trans=(theta_0_max)/C_max
    V_trans=np.roots([c_true[MZI_no],b_true[MZI_no],a_true[MZI_no],-P_trans]) #Don't forget the minus sign!
    filtered = [i.real for i in V_trans if i.imag == 0]
    #print(filtered)
    V_trans=filtered[0] #hopefully there will only be one non complex number

    #return A_max,B_max,C_max,theta_0_max, V_trans,V_ref
    return A_max,B_max,C_max,theta_0_max, V_ref,V_trans

time_start=time.time()
#Block of explicitly working through PIC chip
#Main diagonal
#MZI 1
A_res[0],B_res[0],C_res[0],theta0_res[0],V0_res[0],Vpi_res[0]=curvefit(0,0,3)
V_global[0]=V0_res[0]
#MZI 3
A_res[2],B_res[2],C_res[2],theta0_res[2],V0_res[2],Vpi_res[2]=curvefit(2,0,3)
V_global[2]=V0_res[2]
#MZI 5
A_res[4],B_res[4],C_res[4],theta0_res[4],V0_res[4],Vpi_res[4]=curvefit(4,0,3)
V_global[4]=V0_res[4]

#First upper diagonal
V_global[0]=Vpi_res[0]
#MZI 4
A_res[3],B_res[3],C_res[3],theta0_res[3],V0_res[3],Vpi_res[3]=curvefit(3,0,2)
V_global[3]=V0_res[3]
#MZI 6
A_res[5],B_res[5],C_res[5],theta0_res[5],V0_res[5],Vpi_res[5]=curvefit(5,0,2)
V_global[5]=V0_res[5]


#First lower diagonal
#MZI 2
A_res[1],B_res[1],C_res[1],theta0_res[1],V0_res[1],Vpi_res[1]=curvefit(1,2,3)
V_global[1]=V0_res[1]

time_end=time.time()

#Results

print("time taken")
print(time_end-time_start)
print("C comparison")
print(C_res-C_true)

print("theta_0 comparison")
print(theta0_res-theta0_true)

print("Fidelity comparison")

MatArray_test_true=[]

V_test=np.random.uniform(low=0, high=V_max, size=(n_MZI,N,))

for i in range(N):
    #V_global[MZI_no]=V_test[i]
    P_elect_true = np.multiply(a_true[:,None],V_test) + np.multiply(b_true[:,None],V_test**2) + np.multiply(c_true[:,None],V_test**3) # electrical power
    phi_true=C_true[:,None]*P_elect_true-theta0_true[:,None]

    modeU=np.eye(m,dtype=np.complex128)
    for j in range(len(order)):
        modeU=modeU@globalU(idealMZI(phi_true[order[j],:][i]),m,positions[j])
    MatArray_test_true.append(modeU)

MatArray_test_res=[]

for i in range(N):
    #V_global[MZI_no]=V_test[i]
    P_elect_true = np.multiply(a_true[:,None],V_test) + np.multiply(b_true[:,None],V_test**2) + np.multiply(c_true[:,None],V_test**3) # electrical power
    phi_res=C_res[:,None]*P_elect_true-theta0_res[:,None]

    modeU=np.eye(m,dtype=np.complex128)
    for j in range(len(order)):
        modeU=modeU@globalU(idealMZI(phi_res[order[j],:][i]),m,positions[j])
    MatArray_test_res.append(modeU)

Fid=[]
for _ in range(len(MatArray_test_res)):
    #Fidelity
    Fid.append((1/m)*np.absolute(np.trace(MatArray_test_res[_].conj().T@MatArray_test_true[_])))
    #HS norm
    #Fid.append(np.linalg.norm(MatArray_test_true[_]-MatArray_test_res[_],ord=2))
    #Custom rotation metric
    #Fid.append(np.arccos((np.trace(MatArray_test_true[_].conj().T@MatArray_test_res[_]))/(2*np.sqrt(np.linalg.det(MatArray_test_true[_].conj().T@MatArray_test_res[_])))))

print(Fid)
print("Average Fidelity")
print(np.sum(Fid)/len(Fid))


#print(MatArray_test_res[0])
#print(MatArray_test_true[0])

print("MSE comparison")
print(MSE)
print(sum(MSE))

print("euclid dist")
a=np.stack([C_true-C_res,theta0_true-theta0_res])
eucliddist=np.linalg.norm(a,axis=0)
print(eucliddist[0])