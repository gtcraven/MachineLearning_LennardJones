import sys
import os
import numpy as np
import time
import warnings
from sklearn.kernel_ridge import KernelRidge
pathname = os.path.dirname(sys.argv[0])
fullpath = os.path.abspath(pathname)
warnings.filterwarnings("ignore",category = DeprecationWarning)
warnings.filterwarnings("ignore",category = UserWarning)
warnings.filterwarnings("ignore")
#################################################################


#The input state point. This is the point where the RDF would like to be predicted
rhotarget = 1.0
Ttarget = 2.0

rhoc = 0.316 #LJ critical density
Tc = 1.3262 #LJ critical temperature
rhotp = 0.86 #LJ triple point density

def rho1(T):
    return rhoc + 0.477*(Tc-T)**(1./3.)+0.2124*(Tc-T)-0.0115*(Tc-T)**(1.5)

def rho2(T):
    return rhoc - 0.477*(Tc-T)**(1./3.)+0.05333*(Tc-T)+0.1261*(Tc-T)**(1.5)

def rhoL(T):
    AS = 2.254
    BS = 0.71065
    return np.sqrt((BS+np.sqrt(BS+4.*AS*T))/(2.*AS))

rhofolders = [0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009, \
0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, \
0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, \
0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, \
0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, \
0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, \
0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, \
0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, \
0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, \
0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, \
1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, \
1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2, 1.21, 1.22, \
1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.3, 1.31, 1.32, 1.33, \
1.34, 1.35, 1.36, 1.37, 1.38, 1.39, 1.4, 1.41]

Tfolders = [0.4,0.6,0.8,1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, \
1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, \
2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, \
2.85, 2.9, 2.95, 3.0,\
3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, \
4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4]

pairlist = []

for rho in rhofolders:
    for T in Tfolders:
        isFile = os.path.isfile(os.path.abspath(pathname)+"/Gottschalk_Data/"+ str(rho) +"_" +str(T)+ "_PU.txt") 
        if isFile==True:
            pairlist.append((rho,T))

compute=True

if Ttarget>6.0 or rhotarget>1.25:
    compute = False    
else:    
    if rhotarget>rhotp:
        if rhotarget>rhoL(Ttarget):
            compute=False
    elif rhotarget<rhotp:
        if Ttarget<Tc:
            if rhotarget<rhoc:
                if rhotarget> rho2(Ttarget):
                    compute=False
            elif rhotarget>rhoc:
                if rhotarget<rho1(Ttarget):
                    compute=False

if compute == False: 
    print("The selected state point is either outside of the Lennard-Jones \
fluid region or extrapolation outside of the training data must be used to \
calculate the RDF at this point. Select another state point.")


if compute == True:

    inputlist = []
    Plist = []
    Ulist = []
    
    distarray = []
    for neighbor in pairlist:
        dist = np.sqrt((neighbor[0]-rhotarget)**2.+0.035*(neighbor[1]-Ttarget)**2.)
        distarray.append((neighbor[0],neighbor[1],dist))

    distarray = sorted(distarray, key=lambda x: x[-1])  
    neighborlist = distarray[0:16]
            
    for neighbor in neighborlist:
        rho = neighbor[0]
        T = neighbor[1]
        isFile = os.path.isfile(os.path.abspath(pathname)+"/Gottschalk_Data/"+str(rho)+"_"+str(T)+"_PU.txt")  
        if isFile==True:
            data = np.loadtxt(os.path.abspath(pathname)+"/Gottschalk_Data/"+str(rho)+"_"+str(T)+"_PU.txt")
            Plist.append(data[2])
            Ulist.append(data[3])
            inputpair = [rho, T]
            inputlist.append(inputpair)
        else:
            None

    c0array = np.logspace(-1,7,num = 5)
    error_holdP = 100
    error_holdU = 100
    for c0 in c0array:                
       errorP = 0
       errorU = 0
       for i in range(0,len(inputlist)):
           inp = inputlist[i]
           rhorho = inp[0]
           TT = inp[1]
           
           krrP = KernelRidge(kernel='poly', degree=4, coef0=c0, alpha=1, gamma=100)
           krrP.fit(inputlist,Plist)
           inputout = []
           inputout.append([rhorho,TT])
           inputout_array = np.array(inputout)
           krrP_pred = krrP.predict(inputout_array)
           Ppredicted = krrP_pred
                                   
           krrU = KernelRidge(kernel='poly', degree=4, coef0=c0, alpha=1, gamma=100)
           krrU.fit(inputlist,Ulist)
           inputout = []
           inputout.append([rhorho,TT])
           inputout_array = np.array(inputout)
           krrU_pred = krrU.predict(inputout_array)
           Upredicted = krrU_pred
           
           #print(Ppredicted[0],Plist[i])
           errorP += np.abs((Ppredicted[0]-Plist[i])/Plist[i])*100
           errorU += np.abs((Upredicted[0]-Ulist[i])/Ulist[i])*100
   
       if errorP<error_holdP:
           error_holdP = errorP
           c0minP = c0
           
       if errorU<error_holdU:
           error_holdU = errorU
           c0minU = c0
    

    krrP = KernelRidge(kernel='poly',degree=4, alpha=1, coef0=c0minP, gamma=100)
    krrP.fit(inputlist,Plist)                  
                                              
    krrU = KernelRidge(kernel='poly',degree=4, alpha=1, coef0=c0minU, gamma=100)
    krrU.fit(inputlist,Ulist)

    inputout = []
    inputout.append([rhotarget,Ttarget])
    inputout_array = np.array(inputout) 
    krrP_pred = krrP.predict(inputout_array)
    Ppredicted = krrP_pred
    krrU_pred = krrU.predict(inputout_array)
    Upredicted = krrU_pred

    
    
print("rho =", rhotarget, "T =", Ttarget,"Predicted Pressure =", round(Ppredicted[0],5),"Predicted Internal Energy =", round(Upredicted[0],6))



