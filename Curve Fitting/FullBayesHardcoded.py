"""
Code file of hardcoding a full characterisation, using Dave's characterisation of an MZI but with the overarching process from Niko di Giano thesis.

- A rare case is what if any of the other following MZIs along a diagonal when calibrating are already set to transmission point, then power at output will be zero
- Dave's fitting to cosine comes from matrix expression, frankly the parameters are mostly meaningless except for knowing the transmission and reflection point ('A' scales based on power at output which just depends on routing settings of the other MZIs)
- If the Acos(phi)+B is to mirror the unitary transformation then surely the parameters will be a function of phi too?
- What does beamsplitter imbalance do to characterisation protocol
-Getting mode of posterior didn't seem to help, even worsened slightly
"""

#Import modules
import numpy as np
import multiprocessing
import arviz as az
import pymc as pm
import scipy
import matplotlib.pyplot as plt

#Testing rotation metric
#X=np.array([[0,1],[1,0]],dtype=np.complex128)
#I=np.array([[1,0],[0,1]],dtype=np.complex128)
#print(np.arccos((np.trace(X.conj().T@I))/(2*np.sqrt(np.linalg.det(X.conj().T@I)))))

#Priming
cpucount=multiprocessing.cpu_count()
RANDOM_SEED = 0
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

#Define key parameters and arrays
m=6
N=1000

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
C_true+=1+rng.normal(scale=.1, size=(n_MZI,))
#print(C_true)
theta0_true=np.zeros((n_MZI,))
theta0_true+=rng.normal(scale=.1, size=(n_MZI,))


order=[1,6,2,7,14,8,13,9,12,5,11,4,10,3,0] #order of MZIs to multiply to get final unitary
positions=[3,1,4,2,0,3,1,4,2,0,3,1,4,2,0] #waveguide each MZI is on

V = np.random.uniform(low=0, high=V_max, size=(N,))  # without noise, value we believe we are setting
V_noisy = V + rng.normal(scale=0.05, size=(N,))  # extra noise on voltage, value actually set

def idealMZI(phi): #assume both 50:50 beamsplitters
    globalfactor=np.exp(1j*(phi/2 + np.pi/2))
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
    
    #import matplotlib.pyplot as plt

    #plt.plot(V,clickdata,'s')
    #plt.show()

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
    
    print(A_max)
    print(B_max)
    print(C_max)
    print(theta_0_max)

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

    return A_max,B_max,C_max,theta_0_max, V_trans,V_ref

#Block of explicitly working through PIC chip
#Main diagonal
#MZI 1
A_res[0],B_res[0],C_res[0],theta0_res[0],V0_res[0],Vpi_res[0]=curvefit(0,0,5)
V_global[0]=V0_res[0]
#MZI 5
A_res[4],B_res[4],C_res[4],theta0_res[4],V0_res[4],Vpi_res[4]=curvefit(4,0,5)
V_global[4]=V0_res[4]
#MZI 13
A_res[12],B_res[12],C_res[12],theta0_res[12],V0_res[12],Vpi_res[12]=curvefit(12,0,5)
V_global[12]=V0_res[12]
#MZI 9
A_res[8],B_res[8],C_res[8],theta0_res[8],V0_res[8],Vpi_res[8]=curvefit(8,0,5)
V_global[8]=V0_res[8]
#MZI 3
A_res[2],B_res[2],C_res[2],theta0_res[2],V0_res[2],Vpi_res[2]=curvefit(2,0,5)
V_global[2]=V0_res[2]

#First upper diagonal
V_global[0]=Vpi_res[0]
#MZI 6
A_res[5],B_res[5],C_res[5],theta0_res[5],V0_res[5],Vpi_res[5]=curvefit(5,0,4)
V_global[5]=V0_res[5]
#MZI 14
A_res[13],B_res[13],C_res[13],theta0_res[13],V0_res[13],Vpi_res[13]=curvefit(13,0,4)
V_global[13]=V0_res[13]
#MZI 8
A_res[7],B_res[7],C_res[7],theta0_res[7],V0_res[7],Vpi_res[7]=curvefit(7,0,4)
V_global[7]=V0_res[7]
#MZI 2
A_res[1],B_res[1],C_res[1],theta0_res[1],V0_res[1],Vpi_res[1]=curvefit(1,0,4)
V_global[1]=V0_res[1]

#Second upper diagonal
V_global[5]=Vpi_res[5]
#MZI 15
A_res[14],B_res[14],C_res[14],theta0_res[14],V0_res[14],Vpi_res[14]=curvefit(14,0,2)
V_global[14]=V0_res[14]
#MZI 7
A_res[6],B_res[6],C_res[6],theta0_res[6],V0_res[6],Vpi_res[6]=curvefit(6,0,2)
V_global[6]=V0_res[6]

#First lower diagonal
V_global[2]=Vpi_res[2]
#MZI 4
A_res[3],B_res[3],C_res[3],theta0_res[3],V0_res[3],Vpi_res[3]=curvefit(3,2,5)
V_global[3]=V0_res[3]
#MZI 12
A_res[11],B_res[11],C_res[11],theta0_res[11],V0_res[11],Vpi_res[11]=curvefit(11,2,5)
V_global[11]=V0_res[11]
#MZI 10
A_res[9],B_res[9],C_res[9],theta0_res[9],V0_res[9],Vpi_res[9]=curvefit(9,2,5)
V_global[9]=V0_res[9]

#Second lower diagonal
V_global[9]=Vpi_res[9]

#MZI 11
A_res[10],B_res[10],C_res[10],theta0_res[10],V0_res[10],Vpi_res[10]=curvefit(10,4,5)
V_global[10]=V0_res[10]

#Results

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
    Fid.append((1/6)*np.absolute(np.trace(MatArray_test_res[_].conj().T@MatArray_test_true[_])))
    #HS norm
    #Fid.append(np.linalg.norm(MatArray_test_true[_]-MatArray_test_res[_],ord=2))
    #Custom rotation metric
    #Fid.append(np.arccos((np.trace(MatArray_test_true[_].conj().T@MatArray_test_res[_]))/(2*np.sqrt(np.linalg.det(MatArray_test_true[_].conj().T@MatArray_test_res[_])))))

print(Fid)
print("Average Fidelity")
print(np.sum(Fid)/len(Fid))


#print(MatArray_test_res[0])
#print(MatArray_test_true[0])