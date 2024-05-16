"""
Code to just do MLE like Emilien suggested, given MLE is just bayesian inference with uniform priors, I am not optimistic this will work long term but still

MLE is effectively search to find best parameters that fits the data. Is it just plug parameters into multinomial likelihood and mean square error, but then this is more just an optimisation, how choose next step relative to current?

https://blog.jakuba.net/maximum-likelihood-for-multinomial-distribution/

pi_i=prob of click in mode, x_i=number of clicks in mode, n=total number of clicks
which is common sense, so MLE is parameters that generate the expected probabilities but then the whole point is that there are multiple parameters that generate the same output probability!

Current code very muddled due to picking from different old codes, ultimately still no identifiability because MLE  is doing the same thing where even for a phase shifter then beamsplitter, the phase shifter reconfigurability allows alternate configs that fit and thus lack of repeatability in converged solutions from MLE. Classical light doesn't change this, encoding the pi shift to fit the phase shifter params is what Emilien is really after to help resolve identifiability. This means multistep characterisation, characterise phase shifters first then complete circuit, although the reconfigurability of different splitting ratios for bigger circuits is also an open point.
"""
"""
#Import modules
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize
import scipy.stats
import multiprocessing
from collections import defaultdict
from matplotlib import cbook, cm
from matplotlib.colors import LightSource

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
    
def removekey(d, key): #For removing dictionary keys, may not be needed
    r = dict(d)
    del r[key]
    return r
 
def construct_PS(phi): #Phase shifter component unitary (only valid for 2 modes)
    mat=np.array(([np.exp(1j*phi/2),0],[0,np.exp(-1j*phi/2)]))
    return mat

def construct_BS(eta): #Beamsplitter component unitary
    mat=np.array(([np.sqrt(eta), 1j*np.sqrt(1-eta)],[1j*np.sqrt(1-eta),np.sqrt(eta)]))
    return mat
    

circuit_ordering=[('BS',0.5,1),('PS',[0,0.7],1),('PS',[0.1,0.5],2),('BS',0.49,1)] #Inputting the circuit arrangement (component key, value(s), (top) mode it acts on)
#circuit_ordering=[('BS',0.5,1),('PS',[0,0.7],2),('BS',0.5,1),('PS',[0.1,0.5],2),('BS',0.5,1)] #Inputting the circuit arrangement (component key, value(s), (top) mode it acts on)
circuit = defaultdict(list)
totalorder=[]
for k, v, n in circuit_ordering:
    circuit[k].append(v)
    totalorder.append(k)

#print(circuit)
#print(totalorder)

#print(np.eye(m)@construct_BS(0.5))
#Have to read totalorder backwards for construction of the unitary

#Since each phase shifter has its own set of voltages applied to each other the block of code below makes one array for each phase shifter.
V=[]
for _ in range(len(circuit['PS'])):
    Velem=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5 for phase shifter
    #Velem=Velem+rng.normal(scale=0.02, size=N) #Adding gaussian noise
    V.append(list(Velem))
 
#Since each phase shifter phi has its own set of voltage settings in the experiment I need to track this
expanded_dict= circuit.copy()
expanded_dict['V'].append(V) #exppanded_dict has circuit parameters and the voltages as another key
    
def gen_constructU(etas,phis,m,circuit_ordering): #Function to generate Unitary, indexing is negative because need to move from right to left through circuit (reverse order to circuit input)
    U=np.eye(m,dtype=complex) #Identity transformation to start
    BS_counter=-1 #track beamsplitter value to use as scale to more beamsplitters
    PS_counter=-1 #track phase shifter value to use as scale to more phase shifters
    #print(U)
    for i in range(len(circuit_ordering)):
        if circuit_ordering[-(i+1)][0]=='BS': #Check component key
            index=circuit_ordering[-(i+1)][2]-1 #Find mode index
            newU=np.eye(m,dtype=complex) #start from identity
            #newU[index:index+construct_BS(etas[BS_counter]).shape[0],index:index+construct_BS(etas[BS_counter]).shape[1]]=construct_BS(etas[BS_counter]) #Substitute component unitary
            newU=construct_BS(etas[BS_counter])
            U=U@newU #Multiply out
        if circuit_ordering[-(i+1)][0]=='PS':
            index=circuit_ordering[-(i+1)][2]-1
            newU=np.eye(m,dtype=complex)
            #print(newU[index:index+construct_PS(phis[PS_counter]).shape[0]][index:index+construct_PS(phis[PS_counter]).shape[0]])
            #print(construct_PS(phis[PS_counter]))
            #newU[index:index+construct_PS(phis[PS_counter]).shape[0],index:index+construct_PS(phis[PS_counter]).shape[1]]=construct_PS(phis[PS_counter])
            newU=construct_PS(phis[PS_counter])
            U=U@newU
        #print(U)
    return U
    
def constructU_from_p(eta,phis):
    U=np.eye(2)
    #print(etas)
    #print(phis)
    BS_counter=-1
    PS_counter=-1
    for i in range(len(totalorder)):
        if totalorder[-i]=='BS':
            #U=U@construct_BS(etas[BS_counter])
            U=U@construct_BS(eta)
            BS_counter-=1
        if totalorder[-i]=='PS':
            #U=U@construct_PS(phis[PS_counter])
            U=U@construct_PS(phis)
            PS_counter-=1
    return U



#Data Generation

#should generalise datagen to track input state but will stick with fixed input for now
#Remember top ket extracts leftmost column, then top bra extracts top element
def DataGen(InputNumber,poissonian=False, **expanded_dict):
    data=np.empty((N,m),dtype=complex)
    C=np.empty(N)
    P=np.empty((N,m))
    for i in range(N):
        phis=[]
        for j in range(len(expanded_dict['PS'])):
            phis.append(expanded_dict['PS'][j][0]+expanded_dict['PS'][j][1]*expanded_dict['V'][0][j][i]**2)
        #print(phis)
        #print(expanded_dict['BS'])
        U_true=gen_constructU(expanded_dict['BS'],phis, m, circuit_ordering)
        #print(U_true)
        P_click1_true=abs(U_true[0][0])**2 #Probability of click in top
        P_click1_true=P_click1_true
        P_click2_true=abs(U_true[1][0])**2 #Probability of click in bottom
        P_click2_true=P_click2_true
        P_true=[P_click1_true,P_click2_true]
        P[i]=P_true
        #n=C,p=P,x=array of clicks
        data[i]=scipy.stats.multinomial.rvs(n=InputNumber,p=P_true)
        #Need to add poissonian noise
        if poissonian==True:
            data[i]+=rng.poisson(size=len(data[i]))
        C[i]=np.sum(data[i])
    return data,C,P
    
data,C,P=DataGen(InputNumber=1000,poissonian=False, **expanded_dict)
#print(P)  
L_dev=4E-9 #delta L, chosen so it coincides with Alex's sd to be ~np.pi/200
wv=1550E-9
a_dev=(2*np.pi*L_dev)/wv

#Now I have an array of P_true so now I just need to define objective func (residuals of output probs from parameters and ones produced by data) scipy.optimise.minimize

def costfunc(x):
    #print(x)
    eta1=x[0]
    eta2=x[1]
    a1=x[2]
    a2=x[3]
    b1=x[4]
    b2=x[5]

    P_test=np.empty((N,m))
    for i in range(N):
        #phis=[]
        #for j in range(len(expanded_dict['PS'])):
        #    phis.append(expanded_dict['PS'][j][0]+expanded_dict['PS'][j][1]*expanded_dict['V'][0][j][i]**2)
        etas=[eta1,eta2]
        phis=[a1+b1*V[0][i]**2,a2+b2*V[1][i]**2]
        #phi=a+b*V[0][i]**2
        #U_true=gen_constructU(eta,phis, m, circuit_ordering)        
        U=gen_constructU(etas,phis,m,circuit_ordering)
        P_click1_true=abs(U[0][0])**2 #Probability of click in top
        P_click1_true=P_click1_true
        P_click2_true=abs(U[1][0])**2 #Probability of click in bottom
        P_click2_true=P_click2_true
        P_true=[P_click1_true,P_click2_true]
        P_test[i]=P_true
    #print(P_test)
    res=0
    for _ in range(N):
        #res+=(1/N)*(P[_]-P_test[_])**2 #want norm of residuals as my obj func to minimise
        res+=scipy.linalg.norm(P[_]-P_test[_],ord=2)
    #print(res)
    return res

result=scipy.optimize.minimize(fun=costfunc,x0=np.array([0.5,0.55,0.45,0,0,0.7,0.5]),bounds=[(0,1),(0,1),(0,1),(-np.pi,np.pi),(-np.pi,np.pi),(-np.inf,np.inf),(-np.inf,np.inf)])
#result=scipy.optimize.minimize(fun=costfunc,x0=np.array([0.6,0.55,0.45,0.5,0.3,0.6,0.5]),bounds=[(0,1),(0,1),(0,1),(-np.pi,np.pi),(-np.pi,np.pi),(-np.inf,np.inf),(-np.inf,np.inf)])

print(result)

#plotting_eta=np.linspace(0,1,100)
plotting_eta1=np.linspace(0,1,100)
plotting_eta2=np.linspace(0,1,100)
plotting_a1=np.linspace(-np.pi,np.pi,100)
plotting_a2=np.linspace(-np.pi,np.pi,100)
plotting_b1=np.linspace(0,1,100)
plotting_b2=np.linspace(0,1,100)
plotting_res=np.zeros((100,100))

for i in range(100):
    for j in range(100):
        #comparing a1 and a2
        #x_test=np.array([0.5,0.49,plotting_a1[i],plotting_a2[j],0.7,0.5])
        #comparing eta1 and eta2
        #x_test=np.array([plotting_eta1[i],plotting_eta2[j],0,0.1,0.7,0.5])
        #comparing eta1 and a1
        #x_test=np.array([plotting_eta1[i],0.49,plotting_a1[j],0.1,0.7,0.5])
        #comparing eta2 and a2
        #x_test=np.array([0.5,plotting_eta2[i],0.45,0,plotting_a2[j],0.7,0.5])
        #comparing a1 and b1
        #x_test=np.array([0.5,0.49,plotting_a1[i],0.1,plotting_b1[j],0.5])
        #comparing a1 and b1
        x_test=np.array([0.5,0.49,0,plotting_a2[i],0.7,plotting_b2[j]])
        
        plotting_res[i][j]=costfunc(x_test)

#x, y = np.meshgrid(plotting_a1, plotting_a2)
#x, y = np.meshgrid(plotting_eta1, plotting_eta2)
#x, y = np.meshgrid(plotting_eta1, plotting_a1)
#x, y = np.meshgrid(plotting_eta2, plotting_a2)
#x, y = np.meshgrid(plotting_a1, plotting_b1)
x, y = np.meshgrid(plotting_eta2, plotting_a2)

#region = np.s_[5:50, 5:50]
region=np.s_[0:,0:]
x, y, z = x[region], y[region], plotting_res[region]

# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

plt.show()
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
    
def constructU(eta1,eta2,phi1,phi2):
    mat=construct_PS(phi2)@construct_BS(eta2)@construct_PS(phi1)@construct_BS(eta1)
    #mat=construct_BS(eta2)@construct_PS(phi1)@construct_BS(eta1)
    return mat

eta1=0.5
eta2=0.5
a1=0
a2=0.1
b1=0.7
b2=0.6

#n_phaseshifters=len(a) if isinstance(a, collections.abc.Iterable) else 1
n_phaseshifters=2

#Since each phase shifter has its own set of voltages applied to each other the block of code below makes one array for each phase shifter.
V=[]
for _ in range(n_phaseshifters):
    Velem=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5 for phase shifter
    #print(Velem)
    #Velem=Velem+rng.normal(scale=0.02, size=N) #Adding gaussian noise
    V.append(list(Velem))
 
#print(V)
#Data Generation

#should generalise datagen to track input state but will stick with fixed input for now
#Remember top ket extracts leftmost column, then top bra extracts top element

power=1000 #Input laser
inputmode=0
#print(V[0])

P=np.empty((N,m))
for i in range(N):
    #phis=[]
    #for j in range(n_phaseshifters):
    #    phis.append(a+b*V**2)
    phi1=a1+b1*V[0][i]**2
    phi2=a2+b2*V[1][i]**2
    #print(phi)
    U_true=constructU(eta1,eta2,phi1,phi2)
    #print(U_true)        
    P_click1_true=abs(U_true[0][0])**2 #Probability of click in top
    P_click1_true=P_click1_true
    P_click2_true=abs(U_true[1][0])**2 #Probability of click in bottom
    P_click2_true=P_click2_true
              
    P_true=[P_click1_true,P_click2_true] #power just be getting clicks out multinomially anyway so no point doing multinomial and just indexing out Unitary like expected
    P[i]=P_true



L_dev=4E-9 #delta L, chosen so it coincides with Alex's sd to be ~np.pi/200
wv=1550E-9
a_dev=(2*np.pi*L_dev)/wv

#Now I have an array of P_true so now I just need to define objective func (residuals of output probs from parameters and ones produced by data) scipy.optimise.minimize

def costfunc(x):
    #print(x)
    eta1=x[0]
    eta2=x[1]
    a1=x[2]
    a2=x[3]
    b1=x[4]
    b2=x[5]

    P_test=np.empty((N,m))
    for i in range(N):
        #phis=[]
        #for j in range(len(expanded_dict['PS'])):
        #    phis.append(a+b*np.square(V[]))
        phi=a1+b1*V[0][i]**2
        phi2=a2+b2*V[1][i]**2
        #U_true=gen_constructU(eta,phis, m, circuit_ordering)        
        U=constructU(eta1,eta2,phi1,phi2)
        P_click1_true=abs(U[0][0])**2 #Probability of click in top
        P_click1_true=P_click1_true
        P_click2_true=abs(U[1][0])**2 #Probability of click in bottom
        P_click2_true=P_click2_true
        P_true=[P_click1_true,P_click2_true]
        P_test[i]=P_true
    #print(P_test)
    res=0
    for _ in range(N):
        #res+=(1/N)*(P[_]-P_test[_])**2 #want norm of residuals as my obj func to minimise
        res+=scipy.linalg.norm(P[_]-P_test[_],ord=2)
    #print(res)
    return res

result=scipy.optimize.minimize(fun=costfunc,x0=np.array([0.5,0.5,0,0,0.7,0.7]),bounds=[(0,1),(0,1),(-np.pi,np.pi),(-np.pi,np.pi),(-np.inf,np.inf),(-np.inf,np.inf)])
print(result)

#plotting_eta=np.linspace(0,1,100)
#plotting_res=[]
#for _ in range(len(plotting_eta)):
#    x_test=np.array([plotting_eta[_],0,0.7])
#    plotting_res.append(costfunc(x_test))

#plt.plot(plotting_eta,plotting_res)
#plt.show()
"""
plotting_eta1=np.linspace(0,1,100)
plotting_eta2=np.linspace(0,1,100)
plotting_a1=np.linspace(-np.pi,np.pi,100)
plotting_a2=np.linspace(-np.pi,np.pi,100)
plotting_b1=np.linspace(0,1,100)
plotting_b2=np.linspace(0,1,100)
plotting_res=np.zeros((100,100))

for i in range(100):
    for j in range(100):
        #0.5,0.5,0,0.1,0.7,0.6
        #eta1,eta2
        #x_test=np.array([plotting_eta1[i],plotting_eta2[j],0,0.1,0.7,0.6])
        #eta1,a1
        #x_test=np.array([plotting_eta1[i],0.5,plotting_a1[j],0.1,0.7,0.6])
        #eta1,a2
        #x_test=np.array([plotting_eta1[i],0.5,0,plotting_a2[j],0.7,0.6])
        #eta1,b1
        #x_test=np.array([plotting_eta1[i],0.5,0,0.1,plotting_b1[j],0.6])
        #eta1,b2
        #x_test=np.array([plotting_eta1[i],0.5,0,0.1,0.7,plotting_b2[j]])
        #eta2,a1
        #x_test=np.array([0.5,plotting_eta2[i],plotting_a1[j],0.1,0.7,0.6])
        #eta2,a2
        #x_test=np.array([0.5,plotting_eta2[i],0,plotting_a2[j],0.7,0.6])
        #eta2,b1
        #x_test=np.array([0.5,plotting_eta2[i],0,0.1,plotting_b1[j],0.6])
        #eta2,b2
        #x_test=np.array([0.5,plotting_eta2[i],0,0.1,0.7,plotting_b2[j]])
        #a1,a2
        #x_test=np.array([0.5,0.5,plotting_a1[i],plotting_a2[j],0.7,0.6])
        #a1,b1
        x_test=np.array([0.5,0.5,plotting_a1[i],0.1,plotting_b1[j],0.6])
        #a1,b2
        #x_test=np.array([0.5,0.5,plotting_a1[i],0.1,0.7,plotting_b2[j]])
        #a2,b1
        #x_test=np.array([0.5,0.5,0,plotting_a2[i],plotting_b1[j],0.6])
        #a2,b2
        #x_test=np.array([0.5,0.5,0,plotting_a2[i],0.7,plotting_b2[j]])
        #b1,b2
        #x_test=np.array([0.5,0.5,0,0.1,plotting_b1[i],plotting_b2[j]])
        plotting_res[i][j]=costfunc(x_test)

#x, y = np.meshgrid(plotting_eta1, plotting_eta2)
#x, y = np.meshgrid(plotting_eta1, plotting_a1)
#x, y = np.meshgrid(plotting_eta1, plotting_a2)
#x, y = np.meshgrid(plotting_eta1, plotting_b1)
#x, y = np.meshgrid(plotting_eta1, plotting_b2)
#x, y = np.meshgrid(plotting_eta2, plotting_a1)
#x, y = np.meshgrid(plotting_eta2, plotting_a2)
#x, y = np.meshgrid(plotting_eta2, plotting_b1)
#x, y = np.meshgrid(plotting_eta2, plotting_b2)
#x, y = np.meshgrid(plotting_a1, plotting_a2)
x, y = np.meshgrid(plotting_a1, plotting_b1)
#x, y = np.meshgrid(plotting_a1, plotting_b2)
#x, y = np.meshgrid(plotting_a2, plotting_b1)
#x, y = np.meshgrid(plotting_a2, plotting_b2)
#x, y = np.meshgrid(plotting_b1, plotting_b2)


#region = np.s_[5:50, 5:50]
region=np.s_[0:,0:]
x, y, z = x[region], y[region], plotting_res[region]

# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

plt.show()
"""