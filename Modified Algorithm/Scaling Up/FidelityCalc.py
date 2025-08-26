###Import modules

import numpy as np

np.seterr(all="raise")
###Values

##2 mode

#HMC

eta2_HMC_true=[0.5057573936570046, 0.4810426218560039]
eta2_HMC_max=[0.4924053647413822, 0.5046686511764553]
eta2_HMC_mean=[0.5002799468285238, 0.49986424356300097]

a2_HMC_true=[-0.027368867154870618, -0.020471289310816904]
a2_HMC_max=[0.0011921384134418317, 0.0011921384134418317]
a2_HMC_mean=[-0.0006816619290558305, -0.00019975895125738977]

b2_HMC_true=[0.7423584058857516, 0.7626889189885926]
b2_HMC_max=[0.465506925260147, 0.6543525800365104]
b2_HMC_mean=[0.46539475454740076, 0.6559077602570036]

#Metropolis

eta2_met_true=[0.5057573936570046, 0.4810426218560039]
eta2_met_max=[0.8448735629534825, 0.6709110848173214]
eta2_met_mean=[0.7394069015187621, 0.6732330031650947]

a2_met_true=[-0.027368867154870618, -0.020471289310816904]
a2_met_max=[-0.011941738678906778, -0.011941738678906778]
a2_met_mean=[-0.013703786812537195, 0.0009010272267632564]

b2_met_true=[0.7423584058857516, 0.7626889189885926]
b2_met_max=[1.0031142226578602, 0.6426823885874039]
b2_met_mean=[1.0041500241782741, 0.6557195278480047]

##3 mode

#HMC

eta3_HMC_true=[0.5338230366180812, 0.480899552221109, 0.488787053287418, 0.4848875134772465, 0.4812426441669358, 0.43869019041084906]
eta3_HMC_max=[0.4908806027372765, 0.5078421704514426, 0.4888527850774441, 0.4978992243103484, 0.5062945078360442, 0.5007451200263584]
eta3_HMC_mean=[0.5005527257755262, 0.49987871127533673, 0.5017357921801313, 0.5007938084696061, 0.5001130450514522, 0.5007088536826213]

a3_HMC_true=[0.002879885407513577, 0.026247111781882618, -0.000881735423328105, -2.1756200407423234e-05, -0.010796068029753563, -0.0018452858600546608]
a3_HMC_max=[-0.002728061849709293, -0.002728061849709293, -0.002728061849709293, -0.002728061849709293, -0.002728061849709293, -0.002728061849709293]
a3_HMC_mean=[-0.003311059402476372, 0.00048449749284323163, 0.0011313076406718246, -2.7241366117521813e-05, -0.00019365677676630723, 0.0003571808855096456]

b3_HMC_true=[0.7326316498223822, 0.674083029150696, 0.6682337171263599, 0.7282285178114273, 0.6357396661126667, 0.717674763895381]
b3_HMC_max=[0.1991790786293195, 0.4860269983558547, 1.1067399520143102, 0.777022854053743, 0.6492314577509057, 0.7499660262068691]
b3_HMC_mean=[0.19337190087752038, 0.44430522350186363, 1.1092129564842956, 0.7724966246275622, 0.6490420273305986, 0.7490451505566958]

#Metropolis

eta3_met_true=[0.5338230366180812, 0.480899552221109, 0.488787053287418, 0.4848875134772465, 0.4812426441669358, 0.43869019041084906]
eta3_met_max=[0.5875070129722904, 0.5591066240649161, 0.5702441371105742, 0.5435933511437105, 0.43214442095232375, 0.4589213150470885]
eta3_met_mean=[0.6420534056887919, 0.5716488675192247, 0.5724501511046447, 0.562657778015789, 0.4608159817974386, 0.4505524877713304]

a3_met_true=[0.002879885407513577, 0.026247111781882618, -0.000881735423328105, -2.1756200407423234e-05, -0.010796068029753563, -0.0018452858600546608]
a3_met_max=[0.0023224555558962884, 0.0023224555558962884, 0.0023224555558962884, 0.0023224555558962884, 0.0023224555558962884, 0.0023224555558962884]
a3_met_mean=[0.0062348951133632015, -0.0033969725594687893, -0.007177624718767916, 0.006289516820863549, 0.0022806439025965437, 0.0020207082630539323]

b3_met_true=[0.7326316498223822, 0.674083029150696, 0.6682337171263599, 0.7282285178114273, 0.6357396661126667, 0.717674763895381]
b3_met_max=[0.1991790786293195, 0.4860269983558547, 1.1067399520143102, 0.777022854053743, 0.6492314577509057, 0.7499660262068691]
b3_met_mean=[0.7952143611583553, 0.4730981818620534, 0.6848837660270336, 0.7992628149234496, 0.6356947702145811, 0.7541017312449829]

##General

N=100
V_max=5
n_2_PS=2 #2 mode case=1 MZI = 2 phase shifters
n_3_PS=6 #3 mode case =  3 MZI = 6 phase shifters
V_2= np.random.uniform(0,V_max,size=(n_2_PS,N))
V_3= np.random.uniform(0,V_max,size=(n_3_PS,N))
#print(V_2) #yup, shape is right


###Main body
"""
Will get messy, have opted for the patch fix approach out of ease and simplicity due to time, less easy to look at but process is relatively simple, (copy expressions from calculator.ipynb, find and replace relevant stuff e.g. _0 to [0],exp to np.exp, I to 1j...etc)
"""

##Get theta and phi forms, phi depends on V, maybe do this filling out recursively while iterating through voltages...


def UFill(eta,a,b,V,m): #fill accordingly the right true,max,mean, done via args
    theta=[2*np.arccos(np.sqrt(eta[_])) for _ in range(len(eta))]
    U_list=[]
    for _ in range(N):
        phi=[]
        for __ in range(len(a)):
            phi.append(a[__]+b[__]*V[__][_]**2) #phi[0]=a[0]+b[0]*V[0][_]**2,phi[1]=a[1]+b[1]*V[1][_]**2 
        #print(len(phi))
        if m==2:
            #m=2
            Reck_2_A = (np.exp(1j*phi[0])*np.cos(theta[0]/2)*np.cos(theta[1]/2) - np.sin(theta[0]/2)*np.sin(theta[1]/2))*np.exp(1j*phi[1])
            Reck_2_B = 1j*(np.exp(1j*phi[0])*np.sin(theta[0]/2)*np.cos(theta[1]/2) + np.sin(theta[1]/2)*np.cos(theta[0]/2))*np.exp(1j*phi[1])
            Reck_2_C = 1j*(np.exp(1j*phi[0])*np.sin(theta[1]/2)*np.cos(theta[0]/2) + np.sin(theta[0]/2)*np.cos(theta[1]/2))
            Reck_2_D = -np.exp(1j*phi[0])*np.sin(theta[0]/2)*np.sin(theta[1]/2) + np.cos(theta[0]/2)*np.cos(theta[1]/2)

            U=np.array([[Reck_2_A,Reck_2_B],[Reck_2_C,Reck_2_D]])
        if m==3:
            #m=3
            Reck_3_A = ((-(np.exp(1j*phi[0])*np.sin(theta[1]/2)*np.cos(theta[0]/2) + np.sin(theta[0]/2)*np.cos(theta[1]/2))*(np.exp(1j*phi[2])*np.cos(theta[2]/2)*np.cos(theta[3]/2) - np.sin(theta[2]/2)*np.sin(theta[3]/2))*np.exp(1j*phi[3])*np.sin(theta[4]/2) + (np.exp(1j*phi[0])*np.cos(theta[0]/2)*np.cos(theta[1]/2) - np.sin(theta[0]/2)*np.sin(theta[1]/2))*np.exp(1j*phi[1])*np.cos(theta[4]/2))*np.exp(1j*phi[4])*np.cos(theta[5]/2) - ((np.exp(1j*phi[0])*np.sin(theta[1]/2)*np.cos(theta[0]/2) + np.sin(theta[0]/2)*np.cos(theta[1]/2))*(np.exp(1j*phi[2])*np.cos(theta[2]/2)*np.cos(theta[3]/2) - np.sin(theta[2]/2)*np.sin(theta[3]/2))*np.exp(1j*phi[3])*np.cos(theta[4]/2) + (np.exp(1j*phi[0])*np.cos(theta[0]/2)*np.cos(theta[1]/2) - np.sin(theta[0]/2)*np.sin(theta[1]/2))*np.exp(1j*phi[1])*np.sin(theta[4]/2))*np.sin(theta[5]/2))*np.exp(1j*phi[5])
            Reck_3_B = 1j*((-(np.exp(1j*phi[0])*np.sin(theta[0]/2)*np.sin(theta[1]/2) - np.cos(theta[0]/2)*np.cos(theta[1]/2))*(np.exp(1j*phi[2])*np.cos(theta[2]/2)*np.cos(theta[3]/2) - np.sin(theta[2]/2)*np.sin(theta[3]/2))*np.exp(1j*phi[3])*np.sin(theta[4]/2) + (np.exp(1j*phi[0])*np.sin(theta[0]/2)*np.cos(theta[1]/2) + np.sin(theta[1]/2)*np.cos(theta[0]/2))*np.exp(1j*phi[1])*np.cos(theta[4]/2))*np.exp(1j*phi[4])*np.cos(theta[5]/2) - ((np.exp(1j*phi[0])*np.sin(theta[0]/2)*np.sin(theta[1]/2) - np.cos(theta[0]/2)*np.cos(theta[1]/2))*(np.exp(1j*phi[2])*np.cos(theta[2]/2)*np.cos(theta[3]/2) - np.sin(theta[2]/2)*np.sin(theta[3]/2))*np.exp(1j*phi[3])*np.cos(theta[4]/2) + (np.exp(1j*phi[0])*np.sin(theta[0]/2)*np.cos(theta[1]/2) + np.sin(theta[1]/2)*np.cos(theta[0]/2))*np.exp(1j*phi[1])*np.sin(theta[4]/2))*np.sin(theta[5]/2))*np.exp(1j*phi[5])
            Reck_3_C = -(np.exp(1j*phi[2])*np.sin(theta[2]/2)*np.cos(theta[3]/2) + np.sin(theta[3]/2)*np.cos(theta[2]/2))*(np.exp(1j*phi[4])*np.sin(theta[4]/2)*np.cos(theta[5]/2) + np.sin(theta[5]/2)*np.cos(theta[4]/2))*np.exp(1j*(phi[3] + phi[5]))
            Reck_3_D = 1j*((-(np.exp(1j*phi[0])*np.sin(theta[1]/2)*np.cos(theta[0]/2) + np.sin(theta[0]/2)*np.cos(theta[1]/2))*(np.exp(1j*phi[2])*np.cos(theta[2]/2)*np.cos(theta[3]/2) - np.sin(theta[2]/2)*np.sin(theta[3]/2))*np.exp(1j*phi[3])*np.sin(theta[4]/2) + (np.exp(1j*phi[0])*np.cos(theta[0]/2)*np.cos(theta[1]/2) - np.sin(theta[0]/2)*np.sin(theta[1]/2))*np.exp(1j*phi[1])*np.cos(theta[4]/2))*np.exp(1j*phi[4])*np.sin(theta[5]/2) + ((np.exp(1j*phi[0])*np.sin(theta[1]/2)*np.cos(theta[0]/2) + np.sin(theta[0]/2)*np.cos(theta[1]/2))*(np.exp(1j*phi[2])*np.cos(theta[2]/2)*np.cos(theta[3]/2) - np.sin(theta[2]/2)*np.sin(theta[3]/2))*np.exp(1j*phi[3])*np.cos(theta[4]/2) + (np.exp(1j*phi[0])*np.cos(theta[0]/2)*np.cos(theta[1]/2) - np.sin(theta[0]/2)*np.sin(theta[1]/2))*np.exp(1j*phi[1])*np.sin(theta[4]/2))*np.cos(theta[5]/2))
            Reck_3_E = ((np.exp(1j*phi[0])*np.sin(theta[0]/2)*np.sin(theta[1]/2) - np.cos(theta[0]/2)*np.cos(theta[1]/2))*(np.exp(1j*phi[2])*np.cos(theta[2]/2)*np.cos(theta[3]/2) - np.sin(theta[2]/2)*np.sin(theta[3]/2))*np.exp(1j*phi[3])*np.sin(theta[4]/2) - (np.exp(1j*phi[0])*np.sin(theta[0]/2)*np.cos(theta[1]/2) + np.sin(theta[1]/2)*np.cos(theta[0]/2))*np.exp(1j*phi[1])*np.cos(theta[4]/2))*np.exp(1j*phi[4])*np.sin(theta[5]/2) - ((np.exp(1j*phi[0])*np.sin(theta[0]/2)*np.sin(theta[1]/2) - np.cos(theta[0]/2)*np.cos(theta[1]/2))*(np.exp(1j*phi[2])*np.cos(theta[2]/2)*np.cos(theta[3]/2) - np.sin(theta[2]/2)*np.sin(theta[3]/2))*np.exp(1j*phi[3])*np.cos(theta[4]/2) + (np.exp(1j*phi[0])*np.sin(theta[0]/2)*np.cos(theta[1]/2) + np.sin(theta[1]/2)*np.cos(theta[0]/2))*np.exp(1j*phi[1])*np.sin(theta[4]/2))*np.cos(theta[5]/2)
            Reck_3_F = 1j*(np.exp(1j*phi[2])*np.sin(theta[2]/2)*np.cos(theta[3]/2) + np.sin(theta[3]/2)*np.cos(theta[2]/2))*(-np.exp(1j*phi[4])*np.sin(theta[4]/2)*np.sin(theta[5]/2) + np.cos(theta[4]/2)*np.cos(theta[5]/2))*np.exp(1j*phi[3])
            Reck_3_G = -(np.exp(1j*phi[0])*np.sin(theta[1]/2)*np.cos(theta[0]/2) + np.sin(theta[0]/2)*np.cos(theta[1]/2))*(np.exp(1j*phi[2])*np.sin(theta[3]/2)*np.cos(theta[2]/2) + np.sin(theta[2]/2)*np.cos(theta[3]/2))
            Reck_3_H = -1j*(np.exp(1j*phi[0])*np.sin(theta[0]/2)*np.sin(theta[1]/2) - np.cos(theta[0]/2)*np.cos(theta[1]/2))*(np.exp(1j*phi[2])*np.sin(theta[3]/2)*np.cos(theta[2]/2) + np.sin(theta[2]/2)*np.cos(theta[3]/2))
            Reck_3_I = -np.exp(1j*phi[2])*np.sin(theta[2]/2)*np.sin(theta[3]/2) + np.cos(theta[2]/2)*np.cos(theta[3]/2)

            U=np.array([[Reck_3_A,Reck_3_B,Reck_3_C],[Reck_3_D,Reck_3_E,Reck_3_F],[Reck_3_G,Reck_3_H,Reck_3_I]])

        U_list.append(U)
    return U_list


U2_HMC_true=UFill(eta2_HMC_true,a2_HMC_true,b2_HMC_true,V_2,2)
U2_HMC_max=UFill(eta2_HMC_max,a2_HMC_max,b2_HMC_max,V_2,2)
U2_HMC_mean=UFill(eta2_HMC_mean,a2_HMC_mean,b2_HMC_mean,V_2,2)

U2_met_true=UFill(eta2_met_true,a2_met_true,b2_met_true,V_2,2)
U2_met_max=UFill(eta2_met_max,a2_met_max,b2_met_max,V_2,2)
U2_met_mean=UFill(eta2_met_mean,a2_met_mean,b2_met_mean,V_2,2)

U3_HMC_true=UFill(eta3_HMC_true,a3_HMC_true,b3_HMC_true,V_3,3)
U3_HMC_max=UFill(eta3_HMC_max,a3_HMC_max,b3_HMC_max,V_3,3)
U3_HMC_mean=UFill(eta3_HMC_mean,a3_HMC_mean,b3_HMC_mean,V_3,3)

U3_met_true=UFill(eta3_met_true,a3_met_true,b3_met_true,V_3,3)
U3_met_max=UFill(eta3_met_max,a3_met_max,b3_met_max,V_3,3)
U3_met_mean=UFill(eta3_met_mean,a3_met_mean,b3_met_mean,V_3,3)
###Calculation of average fidelity for mean and max point estimates

def Fidelity(U1,U2):
    F=(1/len(U1))*np.trace(np.absolute(U1@U2))
    return F

Fid2_HMC_Max=np.zeros((N,))
Fid2_HMC_Mean=np.zeros((N,))

Fid2_met_Max=np.zeros((N,))
Fid2_met_Mean=np.zeros((N,))

Fid3_HMC_Max=np.zeros((N,))
Fid3_HMC_Mean=np.zeros((N,))

Fid3_met_Max=np.zeros((N,))
Fid3_met_Mean=np.zeros((N,))

for _ in range(N):
    Fid2_HMC_Max[_]=Fidelity(U2_HMC_true[_],U2_HMC_max[_])
    Fid2_HMC_Mean[_]=Fidelity(U2_HMC_true[_],U2_HMC_mean[_])
    Fid2_met_Max[_]=Fidelity(U2_met_true[_],U2_met_max[_])
    Fid2_met_Mean[_]=Fidelity(U2_met_true[_],U2_HMC_mean[_])
    Fid3_HMC_Max[_]=Fidelity(U3_HMC_true[_],U3_HMC_max[_])
    Fid3_HMC_Mean[_]=Fidelity(U3_HMC_true[_],U3_HMC_mean[_])
    Fid3_met_Max[_]=Fidelity(U3_met_true[_],U3_met_max[_])
    Fid3_met_Mean[_]=Fidelity(U3_met_true[_],U3_met_mean[_])

#Iterate through to get all Fidelities

F_ave_2_HMC_max=(1/N)*np.sum(Fid2_HMC_Max)
F_ave_2_HMC_mean=(1/N)*np.sum(Fid2_HMC_Mean)
F_ave_2_met_max=(1/N)*np.sum(Fid2_met_Max)
F_ave_2_met_mean=(1/N)*np.sum(Fid2_met_Mean)
F_ave_3_HMC_max=(1/N)*np.sum(Fid3_HMC_Max)
F_ave_3_HMC_mean=(1/N)*np.sum(Fid3_HMC_Mean)
F_ave_3_met_max=(1/N)*np.sum(Fid3_met_Max)
F_ave_3_met_mean=(1/N)*np.sum(Fid3_met_Mean)

print(F_ave_2_HMC_max)
print(F_ave_2_HMC_mean)

print(F_ave_2_met_max)
print(F_ave_2_met_mean)

print(F_ave_3_HMC_max)
print(F_ave_3_HMC_mean)

print(F_ave_3_met_max)
print(F_ave_3_met_mean)