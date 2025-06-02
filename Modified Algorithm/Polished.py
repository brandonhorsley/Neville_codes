"""
Code file to clear up messy previous code writing to make it effective so that I can then pivot to actually just trying more advanced modelling methods (regularising priors, additional info sources..etc)

Goals with this code:
- PyMC model context box should calculate like matrix vector calculations rather than plugging in the expression I have drawn from outside the code like I've done previously since that symbolic expression is unscalable to compute
- Eliminate trying to model ('external') phase shifters at end of circuit since they aren't captured by the model
- Function to calculate unitary done in the code file (ties into solving point 1)
- Add further results stuff like closeness between inferred U and true U

Active notes:
- May have to adopt my old circuit input function to make the circuit and then maybe I can define a function from that which creates an e.g. Reck scheme, but even then I only really need the unitary at the moment, thec ircuiti nput would just be to track how many of each component there is.
- Am keeping simple for time being, scaling to bigger circuit just means tensoring with larger identity or np.pad
- Phase shifter matrix is probs wrong for arbitrary mode number? Should replace construct_PS to just do phase shift on appropriate number
-When (or if) I figure out pymc linalg expression then I can replace datagen code with that since that may be more efficient
- Mostly fixed but sampler breaks down but I think this is because it cannot handle complex gradients so I need to use vanilla variables or norm of trig expression that was presented ages ago.
- Big idea is that I may have to restucture this code somewhat again since I don't need to reconstruct the whole unitary since all I need is the (real) probabilities that come from it, so rather than focus on constructing the complex-valued unitary, instead focus on constructing the real probability expression, so make 'p_constructor'...etc using the circuit ordering list.
- Need to circumvent complex numbers altogether, may have to restructure from the ground up since complex numbers occur in both BS and PS matrix so maybe I need to maybe return real and imag when I construct a unitary, but will that even work since I am still making the complex matrix first but just returning real and imaginary bits from the construction function? Is there an alternative way to navigate this? I could have a function that returns the real components of BS/PS and another that returnst the imag components (So real for BS would just be sqrt(eta)*I and imag would be be 0 along diag and sqrt(1-eta) on offdiagonal, keeping phase shifter general it would just be to return sin(phi) for real and cos(phi) for imag). Maybe I'll make a v2 for trying these since I have got this code into a basically working state other than this issue. 
- Got code that should work, need to tidy up and do it more properly to get speedups
- fixed faulty Utopreal...etc indexing
"""

#Import modules
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import scipy as sp
import scipy.stats
import pytensor
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import multiprocessing
from collections import defaultdict

def main():
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
    

    circuit_ordering=[('BS',0.5,1),('PS',[0,0.7],1)] #Inputting the circuit arrangement (component key, value(s), (top) mode it acts on)
    circuit = defaultdict(list)
    totalorder=[]
    for k, v, n in circuit_ordering:
        circuit[k].append(v)
        totalorder.append(k)

    #print(circuit)
    #print(totalorder)

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
        U=np.eye(m) #Identity transformation to start
        BS_counter=-1 #track beamsplitter value to use as scale to more beamsplitters
        PS_counter=-1 #track phase shifter value to use as scale to more phase shifters
        for i in range(len(circuit_ordering)):
            if circuit_ordering[-(i+1)][0]=='BS': #Check component key
                index=circuit_ordering[-(i+1)][2]-1 #Find mode index
                newU=np.eye(m) #start from identity
                newU[index:index+construct_BS(etas[BS_counter]).shape[0],index:index+construct_BS(etas[BS_counter]).shape[1]]=construct_BS(etas[BS_counter]) #Substitute component unitary
                U=U@newU #Multiply out
            if circuit_ordering[-(i+1)][0]=='PS':
                index=circuit_ordering[-(i+1)][2]-1
                newU=np.eye(m)
                newU[index:index+construct_PS(phis[PS_counter]).shape[0],index:index+construct_PS(phis[PS_counter]).shape[1]]=construct_PS(phis[PS_counter])
                U=U@newU
        return U
    

    #Data Generation

    #should generalise datagen to track input state but will stick with fixed input for now
    #Remember top ket extracts leftmost column, then top bra extracts top element
    def DataGen(InputNumber,poissonian=False, **expanded_dict):
        data=np.empty((N,m))
        C=np.empty(N)

        for i in range(N):
            phis=[]
            for j in range(len(expanded_dict['PS'])):
                phis.append(expanded_dict['PS'][j][0]+expanded_dict['PS'][j][1]*expanded_dict['V'][0][j][i]**2)
            U_true=gen_constructU(expanded_dict['BS'],phis, m, circuit_ordering)        
            P_click1_true=abs(U_true[0][0])**2 #Probability of click in top
            P_click1_true=P_click1_true
            P_click2_true=abs(U_true[1][0])**2 #Probability of click in bottom
            P_click2_true=P_click2_true
            P_true=[P_click1_true,P_click2_true]
            #n=C,p=P,x=array of clicks
            data[i]=scipy.stats.multinomial.rvs(n=InputNumber,p=P_true)
            #Need to add poissonian noise
            if poissonian==True:
                data[i]+=rng.poisson(size=len(data[i]))
            C[i]=np.sum(data[i])

        return data,C
    
    data,C=DataGen(InputNumber=1000,poissonian=False, **expanded_dict)
    
    L_dev=4E-9 #delta L, chosen so it coincides with Alex's sd to be ~np.pi/200
    wv=1550E-9
    a_dev=(2*np.pi*L_dev)/wv


    #PyMC Model context
    
    #I will manually specify number of etas,as,bs rather than automate since I would need to incorporate a counter to account for uncalibratable external phase shifters (a simple MZI would then mean I would need to declare shape=0 for a and b which would undoubtedly throw errors!)

    def construct_BS_pymc_real(RV):
        #return pt.stack([pm.math.sqrt(RV), 1j*pm.math.sqrt(1-RV), 1j*pm.math.sqrt(1-RV), pm.math.sqrt(RV)]).reshape((2,2))
        return pt.stack([pm.math.sqrt(RV), 0, 0, pm.math.sqrt(RV)])
    
    def construct_BS_pymc_imag(RV):
        #return pt.stack([pm.math.sqrt(RV), 1j*pm.math.sqrt(1-RV), 1j*pm.math.sqrt(1-RV), pm.math.sqrt(RV)]).reshape((2,2))
        return pt.stack([0, pm.math.sqrt(1-RV), pm.math.sqrt(1-RV), 0])
    
    def construct_PS_pymc_real(RV):
        #return pt.stack([pm.math.exp(1j*RV/2),0,0,pm.math.exp(-1j*RV/2)]).reshape((2,2))
        return pm.math.sin(RV)
    
    def construct_PS_pymc_imag(RV):
        #return pt.stack([pm.math.exp(1j*RV/2),0,0,pm.math.exp(-1j*RV/2)]).reshape((2,2))
        return pm.math.cos(RV)

    def complex_matmult(A, B):
        """"Given `A = A + i * Ai` and `B = B + i * Bi` compute the real and imaginary comonents of `AB`"""
        A, Ai = A
        B, Bi = B
        C = A @ B - Ai @ Bi
        Ci = A @ Bi + Ai @ B
        return C, Ci
    
    def complex_matmult_pymc(A, B):
        """"Given `A = A + i * Ai` and `B = B + i * Bi` compute the real and imaginary comonents of `AB`"""
        A, Ai = A
        B, Bi = B
        C = pt.dot(A,B) - pt.dot(Ai, Bi)
        Ci = pt.dot(A, Bi) + pt.dot(Ai, B)
        return C, Ci
    

    def circuit_list_to_matrix_pymc(feature):
        #Had to be quirky in construct_BS_pymc and construct_PS_pymc since set_subtensor doesn't like replacing 2D tensors or rows or even element by element (breaks down after one element) so had to return a flat tensor and reassign half of the array and then the other half.
        index=feature[2]-1
        if feature[0]=="BS":
            real=pt.eye(n=m,m=m)
            real=pt.set_subtensor(pt.flatten(real)[index:index+2],construct_BS_pymc_real(feature[1])[0:2])
            real=pt.set_subtensor(pt.flatten(real)[index+2:index+4],construct_BS_pymc_real(feature[1])[2:4])
            real=real.reshape((2,2))

            imag=pt.eye(n=m,m=m)
            imag=pt.set_subtensor(pt.flatten(imag)[0:2],construct_BS_pymc_imag(feature[1])[0:2])
            imag=pt.set_subtensor(pt.flatten(imag)[2:4],construct_BS_pymc_imag(feature[1])[2:4])
            imag=imag.reshape((2,2))
            #print(real.eval())
            #print(imag.eval())
            matrix=[(real,imag)]*N #To copy it N times to help matrix dotting across to get final U
            #print(matrix)
        if feature[0]=="PS":
            matrix=[]
            for _ in range(N):
                real=pt.eye(n=m,m=m)
                real=pt.set_subtensor(real[index,index],construct_PS_pymc_real(feature[1][0][_]))

                imag=pt.eye(n=m,m=m)
                imag=pt.set_subtensor(imag[index,index],construct_PS_pymc_imag(feature[1][0][_]))

                matrix.append((real,imag))
        #print(feature[0])
        #print(matrix)
        return matrix #return output matrix


    with pm.Model():

        """
        Free parameters to infer
        """
        eta=pm.TruncatedNormal("eta",mu=0.5,sigma=0.05,lower=0.0,upper=1.0,initval=0.5) #array of 
        theta=pm.Deterministic("theta",2*pt.arccos(pt.sqrt(eta)))
        #priors for conciseness
        #sd=a_dev
        a=pm.TruncatedNormal("a", mu=0, sigma=a_dev,lower=-np.pi,upper=np.pi,initval=0)  #array of priors for conciseness
        #sd=0.07
        b=pm.Normal("b", mu=0.7, sigma=0.7,initval=0.7) #array of priors for conciseness
        
        #Volt=pm.Normal("Volt",mu=V_2_dist,sigma=0.1)
        Volt=pm.Deterministic("Volt",pt.as_tensor(V))

        #below expression breaks down when there is just 1 a and b
        #phi=pm.Deterministic("phi",a[:,None]+b[:,None]*pm.math.sqr(Volt))
        """
        phi describes the different phase shifts for different experiments
        """
        phi=pm.Deterministic("phi",a+b*pm.math.sqr(Volt))

        circuit_list=[["BS",eta,1],["PS",phi,1]] #Need to reverse this order for it to be correct
        
        U_list = np.array([circuit_list_to_matrix_pymc(feature) for feature in circuit_list])
        #U_list is an array that I need to dot across but dot via complex_matmul function that I've defined
        #U=pt.nlinalg.matrix_dot(U_list) #Doesn't work raw since PS is a list of N matrices for the N experiments
        
        
        U=[] #To store final mode Unitaries: U=[(U1,U1i),(U2,U2i),...,(UN,UNi)]

        for i in range(N):
            rval = U_list[:,i][0]
            for a in U_list[:,i][1:]:
                rval=complex_matmult_pymc(rval,a)
            U.append(rval)
        
        """
        Indexing specific elements from each array
        """
        Utopreal=[elem[0][0][0] for elem in U] #top left element of each real matrix in U
        Utopimag=[elem[1][0][0] for elem in U] #top left element of each imag matrix in U
        Ubotreal=[elem[0][1][0] for elem in U] #bottom left element of each real matrix in U
        Ubotimag=[elem[1][1][0] for elem in U] #bottom left element of each imag matrix in U

        """
        Big slowdown when attempting to call sampling, text indicating initialisation doesn't even show up
        """
        #P=pm.math.stack([pt.nlinalg.norm(pm.math.stack([Utopreal,Utopimag],axis=-1),ord='fro',axis=-1)**2,pt.nlinalg.norm(pm.math.stack([Ubotreal,Ubotimag],axis=-1),ord='fro',axis=-1)**2])
        P=pm.math.stack([pm.math.sqr(Utopreal)+pm.math.sqr(Utopimag),pm.math.sqr(Ubotreal)+pm.math.sqr(Ubotimag)],axis=-1)
        #print(P.eval()) #Works as expected
        
        likelihood=pm.Multinomial("likelihood",n=C,p=P,shape=(N,m),observed=data)
        
        trace=pm.sample(draws=int(1e3), chains=4, cores=cpucount, return_inferencedata=True)
    
    #Diagnostics/Results
    #Usual Arviz diagnostics
    #Could be good to get a unitary 'closeness' measure like TVD that takes the inference and translates that into an experimentally relevant measure that way
    
if __name__=='__main__':
    main()