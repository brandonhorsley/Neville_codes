"""
This code is to investigate/illustrate identifiability problems from scaling up to bigger circuits, so even by removing the reconfigurability that phase shifters bring, see impact on convergence for just a big chip of beamsplitters, effectively just a series of splitting ratios, intuitively I expect non identifiability even from this too, so ultimately scalable characterisation requires comparative isolation of components and as many info sources as reasonably possible
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
m=3 #Number of circuit modes ('rows' in the circuit)
Vmax=10 #Max voltage setting on phase shifters
N=100 #Top of page 108 ->N=number of experiments

#Supporting functions (keeping simple with dim=2 for now)
    
def removekey(d, key): #For removing dictionary keys, may not be needed
    r = dict(d)
    del r[key]
    return r

#Shouldn't be used
def construct_PS(phi): #Phase shifter component unitary (only valid for 2 modes)
    mat=np.array(([np.exp(1j*phi/2),0],[0,np.exp(-1j*phi/2)]))
    return mat
#shouldn't be used
def construct_BS(eta): #Beamsplitter component unitary
    mat=np.array(([np.sqrt(eta), 1j*np.sqrt(1-eta)],[1j*np.sqrt(1-eta),np.sqrt(eta)]))
    return mat
    
#3 mode
circuit_ordering=[('BS',0.51,1),('BS',0.49,2),('BS',0.55,1)]
#4 mode Clements
#circuit_ordering=[('BS',0.48,1),('BS',0.5,3),('BS',0.51,2),('BS',0.52,1),('BS',0.48,3),('BS',0.5,2)]

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
    #print(U)
    for i in range(len(circuit_ordering)):
        if circuit_ordering[-(i+1)][0]=='BS': #Check component key
            index=circuit_ordering[-(i+1)][2]-1 #Find mode index
            newU=np.eye(m,dtype=complex) #start from identity
            newU[index:index+construct_BS(etas[BS_counter]).shape[0],index:index+construct_BS(etas[BS_counter]).shape[1]]=construct_BS(etas[BS_counter]) #Substitute component unitary
            #newU=construct_BS(etas[BS_counter])
            U=U@newU #Multiply out
        #print(U)
    return U
    
def constructU_from_p(eta,phis):
    U=np.eye(2)
    #print(etas)
    #print(phis)
    BS_counter=-1
    for i in range(len(totalorder)):
        if totalorder[-i]=='BS':
            #U=U@construct_BS(etas[BS_counter])
            U=U@construct_BS(eta)
            BS_counter-=1
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
    #3 mode
    eta1=x[0]
    eta2=x[1]
    eta3=x[2]
    #below include for 4 mode Clements
    eta4=x[3]
    eta5=x[4]
    eta6=x[5]

    P_test=np.empty((N,m))
    for i in range(N):
        phis=[]
        for j in range(len(expanded_dict['PS'])):
            phis.append(expanded_dict['PS'][j][0]+expanded_dict['PS'][j][1]*expanded_dict['V'][0][j][i]**2)
        etas=[eta1,eta2]
        #phis=[a1+b1*V[0][i]**2,a2+b2*V[1][i]**2]
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
#m=2 #Number of circuit modes ('rows' in the circuit)

#Supporting functions (keeping simple with dim=2 for now)

def construct_BS(eta,index,m): #Beamsplitter component unitary
    newU=np.eye(m,dtype=complex)
    index-=1
    #mat=np.array(([np.sqrt(eta), 1j*np.sqrt(1-eta)],[1j*np.sqrt(1-eta),np.sqrt(eta)]))
    newU[index:index+2,index:index+2]=np.array(([np.sqrt(eta), 1j*np.sqrt(1-eta)],[1j*np.sqrt(1-eta),np.sqrt(eta)])) #Substitute component unitary
    return newU
    
def constructU_3(eta1,eta2,eta3):
    mat=construct_BS(eta3,1,3)@construct_BS(eta2,2,3)@construct_BS(eta1,1,3)
    return mat

def constructU_4(eta1,eta2,eta3,eta4,eta5,eta6):
    mat=construct_BS(eta6,2,4)@construct_BS(eta5,3,4)@construct_BS(eta4,1,4)@construct_BS(eta3,2,4)@construct_BS(eta2,3,4)@construct_BS(eta1,1,4)
    return mat

eta1=0.5
eta2=0.5
eta3=0.5
eta4=0.5
eta5=0.5
eta6=0.5

#Data Generation

#should generalise datagen to track input state but will stick with fixed input for now
#Remember top ket extracts leftmost column, then top bra extracts top element

power=1000 #Input laser
inputmode=0
#print(V[0])

#U_true=constructU_3(eta1,eta2,eta3)
U_true=constructU_4(eta1,eta2,eta3,eta4,eta5,eta6)
#print(U_true)        
P_click1_true=abs(U_true[0][0])**2 #Probability of click in top
P_click2_true=abs(U_true[1][0])**2 #Probability of click in bottom
P_click3_true=abs(U_true[2][0])**2 #Probability of click in bottom
P_click4_true=abs(U_true[3][0])**2 #Probability of click in bottom

              
#P_true=np.array([P_click1_true,P_click2_true,P_click3_true]) #power just be getting clicks out multinomially anyway so no point doing multinomial and just indexing out Unitary like expected
P_true=np.array([P_click1_true,P_click2_true,P_click3_true,P_click4_true]) #power just be getting clicks out multinomially anyway so no point doing multinomial and just indexing out Unitary like expected

P=P_true
#print(P)


L_dev=4E-9 #delta L, chosen so it coincides with Alex's sd to be ~np.pi/200
wv=1550E-9
a_dev=(2*np.pi*L_dev)/wv

#Now I have an array of P_true so now I just need to define objective func (residuals of output probs from parameters and ones produced by data) scipy.optimise.minimize

def costfunc(x):
    eta1=x[0]
    eta2=x[1]
    eta3=x[2]
    eta4=x[3]
    eta5=x[4]
    eta6=x[5]
      
    #U=constructU_3(eta1,eta2,eta3)
    U=constructU_4(eta1,eta2,eta3,eta4,eta5,eta6)
    P_click1_true=abs(U[0][0])**2 #Probability of click in top
    P_click2_true=abs(U[1][0])**2 #Probability of click in bottom
    P_click3_true=abs(U[2][0])**2 #Probability of click in top
    P_click4_true=abs(U[3][0])**2 #Probability of click in bottom
    #P_true=[P_click1_true,P_click2_true,P_click3_true]
    P_true=[P_click1_true,P_click2_true,P_click3_true,P_click4_true]
    P_test=np.array(P_true)
    #print(P-P_test)
    res=0
    res+=scipy.linalg.norm(P-P_test,ord=2)
    #print(res)
    return res

#result_3=scipy.optimize.minimize(fun=costfunc,x0=np.array([0.55,0.6,0.48]),bounds=[(0,1),(0,1),(0,1)])
#print(result_3)
result_4=scipy.optimize.minimize(fun=costfunc,x0=np.array([0.7,0.55,0.6,0.49,0.5,0.45]),bounds=[(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)])
print(result_4)

plotting_eta1=np.linspace(0,1,100)
plotting_eta2=np.linspace(0,1,100)
plotting_eta3=np.linspace(0,1,100)
plotting_eta4=np.linspace(0,1,100)
plotting_eta5=np.linspace(0,1,100)
plotting_eta6=np.linspace(0,1,100)
plotting_res=np.zeros((100,100))

for i in range(100):
    for j in range(100):
        #0.5,0.5,0.5,0.5,0.5
        #eta1,eta2
        #x_test=np.array([plotting_eta1[i],plotting_eta2[j],0.5,0.5,0.5,0.5])
        #eta1,eta3
        #x_test=np.array([plotting_eta1[i],0.5,plotting_eta3[j],0.5,0.5,0.5])
        #eta1,eta4
        #x_test=np.array([plotting_eta1[i],0.5,0.5,plotting_eta4[j],0.5,0.5])
        #eta1,eta5
        #x_test=np.array([plotting_eta1[i],0.5,0.5,0.5,plotting_eta5[j],0.5])
        #eta1,eta6
        #x_test=np.array([plotting_eta1[i],0.5,0.5,0.5,0.5,plotting_eta6[j]])
        #eta2,eta3
        #x_test=np.array([0.5,plotting_eta2[i],plotting_eta3[j],0.5,0.5,0.5])
        #eta2,eta4
        #x_test=np.array([0.5,plotting_eta2[i],0.5,plotting_eta4[j],0.5,0.5])
        #eta2,eta5
        #x_test=np.array([0.5,plotting_eta2[i],0.5,0.5,plotting_eta5[j],0.5])
        #eta2,eta6
        #x_test=np.array([0.5,plotting_eta2[i],0.5,0.5,0.5,plotting_eta6[j]])
        #eta3,eta4
        #x_test=np.array([0.5,0.5,plotting_eta3[i],plotting_eta4[j],0.5,0.5])
        #eta3,eta5
        #x_test=np.array([0.5,0.5,plotting_eta3[i],0.5,plotting_eta5[j],0.5])
        #eta3,eta6
        #x_test=np.array([0.5,0.5,plotting_eta3[i],0.5,0.5,plotting_eta6[j]])
        #eta4,eta5
        #x_test=np.array([0.5,0.5,0.5,plotting_eta4[i],plotting_eta5[j],0.5])
        #eta4,eta6
        #x_test=np.array([0.5,0.5,0.5,plotting_eta4[i],0.5,plotting_eta6[j]])
        #eta5,eta6
        x_test=np.array([0.5,0.5,0.5,0.5,plotting_eta5[i],plotting_eta6[j]])
        plotting_res[i][j]=costfunc(x_test)

x, y = np.meshgrid(plotting_eta1, plotting_eta2)
#x, y = np.meshgrid(plotting_eta1, plotting_eta3)
#x, y = np.meshgrid(plotting_eta1, plotting_eta4)
#x, y = np.meshgrid(plotting_eta1, plotting_eta5)
#x, y = np.meshgrid(plotting_eta1, plotting_eta6)
#x, y = np.meshgrid(plotting_eta2, plotting_eta3)
#x, y = np.meshgrid(plotting_eta2, plotting_eta4)
#x, y = np.meshgrid(plotting_eta2, plotting_eta5)
#x, y = np.meshgrid(plotting_eta2, plotting_eta6)
#x, y = np.meshgrid(plotting_eta3, plotting_eta4)
#x, y = np.meshgrid(plotting_eta3, plotting_eta5)
#x, y = np.meshgrid(plotting_eta3, plotting_eta6)
#x, y = np.meshgrid(plotting_eta4, plotting_eta5)
#x, y = np.meshgrid(plotting_eta4, plotting_eta6)
#x, y = np.meshgrid(plotting_eta5, plotting_eta6)


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