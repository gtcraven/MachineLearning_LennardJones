#This program generates the Lennard-Jones Radial Distribution Function (RDF)
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
pathname = os.path.dirname(sys.argv[0])
fullpath = os.path.abspath(pathname)
##############################################

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

rhofolders = [0.005,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,\
              0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25]
Tfolders = [0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,\
            3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0]
pairlist = []

#make a list of all folders that contain an RDF file
for rho in rhofolders:
    for T in Tfolders:
        isFile1 = os.path.isfile(os.path.abspath(pathname)+"/RDF_Data/"+ str(rho) +"/" +str(T)+ "/RDF.txt") 
        if isFile1==True:
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
    
    RDF = []
    RDFnoarray = []
    r = []

    #Find 4 closest points to the input point
    distarray = []
    for pair in pairlist:
        dist = np.sqrt((pair[0]-rhotarget)**2.+0.035*(pair[1]-Ttarget)**2.)
        distarray.append((pair[0],pair[1],dist))

    distarray = sorted(distarray, key=lambda x: x[-1])    

    neighborlist = distarray[0:4]
    
    data = np.loadtxt(os.path.abspath(pathname)+"/RDF_Data/"+ str(neighborlist[0][0]) +"/" +str(neighborlist[0][1])+ "/RDF.txt")     
    r1,RDF1= data[:,0], data[:,1]
    
    data = np.loadtxt(os.path.abspath(pathname)+"/RDF_Data/"+ str(neighborlist[1][0]) +"/" +str(neighborlist[1][1])+ "/RDF.txt")     
    r2,RDF2= data[:,0], data[:,1]
    
    data = np.loadtxt(os.path.abspath(pathname)+"/RDF_Data/"+ str(neighborlist[2][0]) +"/" +str(neighborlist[2][1])+ "/RDF.txt")     
    r3,RDF3= data[:,0], data[:,1] 

    data = np.loadtxt(os.path.abspath(pathname)+"/RDF_Data/"+ str(neighborlist[3][0]) +"/" +str(neighborlist[3][1])+ "/RDF.txt")     
    r4,RDF4= data[:,0], data[:,1]
    
    lowest_number_of_r_points = np.min([len(r1),len(r2),len(r3),len(r4)])
    lowest_end_value_of_r = r1[lowest_number_of_r_points-1] 
    
 
    for i in range(0,lowest_number_of_r_points):
        RDFlist = []
        inputlist = []      
          
        inputpair = [neighborlist[0][0],neighborlist[0][1]]
        inputlist.append(inputpair) 
        RDFlist.append(RDF1[i])
        
        inputpair = [neighborlist[1][0],neighborlist[1][1]]
        inputlist.append(inputpair) 
        RDFlist.append(RDF2[i])
        
        inputpair = [neighborlist[2][0],neighborlist[2][1]]
        inputlist.append(inputpair) 
        RDFlist.append(RDF3[i])
        
        inputpair = [neighborlist[3][0],neighborlist[3][1]]
        inputlist.append(inputpair) 
        RDFlist.append(RDF4[i])
        
        lr = linear_model.LinearRegression()
        lr.fit(inputlist,RDFlist)
        inputout = []
        inputout.append([rhotarget,Ttarget])
        inputout_array = np.array(inputout) 
        lr_pred = lr.predict(inputout_array)
        
        RDFnoarray.append(lr_pred[0])
        RDF.append(lr_pred)
        r.append(r1[i])
                
    print("Finished gathering data and computing the RDF:")
    
    #Write RDF to disk
    R = r[0:lowest_number_of_r_points:1]
    RDF_file=open(os.path.abspath(pathname)+"/ML_RDF_"+ str(rhotarget) + "_" + str(Ttarget) + ".txt", "w") #windows
    
    for k in range(0,len(R)):
        print(R[k],RDFnoarray[k],file = RDF_file)
        
    RDF_file.close()
    
    #Plot the RDF
    plt.figure(figsize=(8,4.75))
    plt.plot(r,RDF,ls = '-',lw=2.0,color = "r")
    plt.ylabel(r'$g(r)$',fontsize=26,labelpad = 6)
    plt.xlabel(r'$r$',fontsize=26,labelpad = 2)
    plt.xlim(0,4.0)
    plt.ylim(-0.2,3.2)    
    plt.show()
