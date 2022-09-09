# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 13:14:34 2016

@author: Александр
"""

import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import sys, os
linewide=2
#Adding a comment here just to test if GitHub connection works

# Loading the data files.
# Spectra in parallel and perpendicular polarization combination and errorbars.
parfile='Parallel_full.dat'
parerrfile='Parallelerror_full.dat'
perfile='Perpendicular_full.dat'
pererrfile='Perpendicularerror_full.dat'

par=np.loadtxt(parfile)[1:,1:]
parerr=np.loadtxt(pererrfile)[1:,1:]
per=np.loadtxt(perfile)[1:,1:]
pererr=np.loadtxt(pererrfile)[1:,1:]

# Creating x and y axes - wavenumbers (frequency) and delay time
wavenumbers=np.loadtxt(parfile)[1:,0]
time=np.loadtxt(parfile)[0,1:]

# Calculating isotropic signal and errorbars
isosignal=(par+2*per)/3
isoerr=(parerr+2*pererr)/3
        
# Plotting raw data for visual check for possible errors in frequency domain
plt.figure(figsize=(10,5))
z=0
times_to_plot=np.array([8,11,14,18,24,28,31,33])
plt.plot(wavenumbers,np.zeros_like(wavenumbers))
color=iter(plt.cm.rainbow(np.linspace(1,0,len(times_to_plot))))
for z in times_to_plot:
    c=next(color)
    plt.plot(wavenumbers,isosignal[:,z],linewidth=2,color=c,label=np.str(time[z])+' ps')  
plt.title('delta A spectra')
plt.xlabel('Wavenumber, cm-1')
plt.ylabel('delta_A')
plt.xlim(np.min(wavenumbers),np.max(wavenumbers))
plt.ylim(-0.013,0.003)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
plt.show()  
    
# Checking visually for errors in time domain
plt.figure(figsize=(10,5))
wvn_to_plot=np.array([25,73])
for i in wvn_to_plot:
    plt.errorbar(time[2:],isosignal[i,2:],yerr=isoerr[i,2:],label=np.str(wavenumbers[i])+' cm-1 ')   
plt.title('delta A spectra')
plt.xlabel('time, ps')
plt.ylabel('delta_A')

plt.xlim(0.26,20)
plt.xticks(np.arange(0,20,2))
plt.grid()
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.show() 

# Calculating anisotropy parameter and anisotropy error
R=(par-per)/(par+2*per)
Rerr=np.abs(R)*np.sqrt(((parerr+pererr)/(par-per))**2+((parerr+2*pererr)/(par+2*per))**2)

# Averaging anisotropy over few ranges and plotting to see the trends 
# in different spectroscopical regions
plt.figure(figsize=(10,5))
R1=(np.sum(par[22:48,:],axis=0)-np.sum(per[22:48,:],axis=0))/(np.sum(par[22:48,:],axis=0)+2*np.sum(per[22:48,:],axis=0))
R2=(np.sum(par[60:80,:],axis=0)-np.sum(per[60:80,:],axis=0))/(np.sum(par[60:80,:],axis=0)+2*np.sum(per[60:80,:],axis=0))
R3=(np.sum(par[102:116,:],axis=0)-np.sum(per[102:116,:],axis=0))/(np.sum(par[102:116,:],axis=0)+2*np.sum(per[102:116,:],axis=0))
R4=(np.sum(par[90:102,:],axis=0)-np.sum(per[90:102,:],axis=0))/(np.sum(par[90:102,:],axis=0)+2*np.sum(per[90:102,:],axis=0))
R5=(np.sum(par[125:138,:],axis=0)-np.sum(per[125:138,:],axis=0))/(np.sum(par[125:138,:],axis=0)+2*np.sum(per[125:138,:],axis=0))
R6=(np.sum(par[142:145,:],axis=0)-np.sum(per[142:145,:],axis=0))/(np.sum(par[142:145,:],axis=0)+2*np.sum(per[142:145,:],axis=0))
plt.plot(time[2:],R1[2:],linewidth=linewide,label='R1')
plt.plot(time[2:],R2[2:],linewidth=linewide,label='R2')
plt.plot(time[2:],R3[2:],linewidth=linewide,label='R3')
plt.plot(time[2:],R4[2:],linewidth=linewide,label='R4')
plt.plot(time[2:],R5[2:],linewidth=linewide,label='R3')
plt.plot(time[2:],R6[2:],linewidth=linewide,label='R6')
plt.title('delta A spectra')
plt.xlabel('time, ps')
plt.ylabel('R')
plt.xlim(0.25,5)
plt.xticks(np.arange(0.5,5,0.5))
plt.grid()
plt.ylim(-0.05,0.25)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
plt.show() 

# Writing the calculated anisotropy into a file for further processing
location=os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
filename='HBr_w12.dat' 
file=os.path.join(location,filename)
result=np.array([time,R1,R2,R3])
np.savetxt(file,result,delimiter=' ')

# Cutting out early delay times for the fitting (to determine rate parameters)
cut=6
t=time[cut:-5]
data=np.transpose(isosignal[:,cut:-5])

# Cutting out early delay times for the spectral signature calculation
cuta=6
ta=time[cuta:-5]
par=np.transpose(par[:,cuta:-5])
per=np.transpose(per[:,cuta:-5])
test=0

# Setting up initial guess for the fit
interm=-1    # when positive - intermediate state (number <interm>) has zero spectral signature
rateparam=np.array([2,0.5,0.5])                
borders=np.array([[0,100],[0,100],[0,1]])


# Setting up weights for each experimental point: either equal or reciprocal of errorbars
isoerr=np.ones_like(np.transpose(isoerr[:,cut:-5]))/1000
#isoerr=np.transpose(isoerr[:,cut:-5])
weights=np.square(np.reciprocal(isoerr))/10000000


# Chi-squared parameter calculation
def chi2(rate):

# Kinetic model matrix (3 or 4 levels)
    pop0=np.array([1,0,0])
    ratem=np.array([[-rate[0],         0,        0],
                    [rate[0],          -rate[1],        0],
                    [0,          rate[1],           0]])
#    ratem=np.array([[-rate[1],        0,        0,      0],
#                     [ rate[1],-rate[2],        0,      0],
#                     [        0, rate[2],-rate[0],      0],
#                     [        0,        0, rate[0],      0]])

    chi2_0=0
    i=0   
    
    # Case 1. Intermediate state has signature
    if interm<0:
        pop=np.empty((len(t),len(pop0)))
        ssign=np.empty((len(pop0),len(wavenumbers)))
        ssignwoi=np.empty_like(ssign)
        
        # Creating populations matrix
        for i in range(len(t)):
            pop[i,:]=np.dot(linalg.expm(ratem*t[i]),pop0)
        
        # Calculating spectral signatures by solving the matrix equation
        i=0
        for i in range(len(wavenumbers)):
            w=np.sqrt(np.diag(weights[:,i]))
            popw=np.dot(w,pop)
            dataw=np.dot(data[:,i],w)
            ssign[:,i]=linalg.lstsq(popw,dataw)[0]
            
    # Case 2. Intermediate state has zero signature    
    else:
        pop=np.empty((len(t),len(pop0)))
        ssign=np.empty((len(pop0),len(wavenumbers)))
        ssignwoi=np.empty((len(pop0)-1,len(wavenumbers)))
        
        # Creating population matrix
        for i in range(len(t)):
            pop[i,:]=np.dot(linalg.expm(ratem*t[i]),pop0)
        
        # Calculating spectral signatures by solving the matrix equation
        i=0
        for i in range(len(wavenumbers)):
            w=np.sqrt(np.diag(weights[:,i]))
            popw=np.dot(w,pop)
            popwoi=np.delete(popw,interm,axis=1)
            dataw=np.dot(data[:,i],w)
            ssignwoi[:,i]=linalg.lstsq(popwoi,dataw)[0]
        ssign=np.insert(ssignwoi,interm,np.zeros(len(wavenumbers)),axis=0)
    
    # Adding custom constraints to add physical meaning to the parameters and signatures
    constraints=10*sum(np.absolute(rate-borders[:,0])-(rate-borders[:,0]))+10*sum(np.absolute(borders[:,1]-rate)-(borders[:,1]-rate))

    # Calculating Chi-squared with added constraints
    chi2_0=sum(sum(((data-np.dot(pop,ssign))/isoerr)**2))/(len(data[0,:])*len(data[:,0]))+constraints

    return chi2_0

# Optimization procedure
q=0
for q in range(5):
    resr=minimize(chi2,rateparam,method='CG', options={'maxiter': 10000})
    print(resr)
    if resr['success']==False:
        rateparam=resr['x']
    else:
        break    

p0=resr['x']
print(resr)

# Calculating spectral signatures using the optimized parameters
# The logic is similar to the Chi-squared calculation procedure
def signatures(rate):


    pop0=np.array([1,0,0])
    ratem=np.array([[-rate[0],         0,        0],
                    [rate[0],          -rate[1],        0],
                    [0,          rate[1],           0]])
#    ratem1=np.array([[-rate1[1],        0,        0,      0],
#                     [ rate1[1],-rate1[2],        0,      0],
#                     [        0, rate1[2],-rate1[0],      0],
#                     [        0,        0, rate1[0],      0]])

    i=0  
    pop1=np.empty((len(t),len(pop0)))
    ssign1=np.empty((len(pop0),len(wavenumbers)))
    for i in range(len(t)):
        pop1[i,:]=np.dot(linalg.expm(ratem*t[i]),pop0)
        
    t1=np.arange(0,5,0.05)
    pop2=np.empty((len(t1),len(pop0)))
    i=0
    for i in range(len(t1)):
        pop2[i,:]=np.dot(linalg.expm(ratem*t1[i]),pop0)
    plt.figure(figsize=(12,7))
    plt.plot(t1,pop2[:,0],linewidth=linewide,label='A')   
    plt.plot(t1,pop2[:,1],linewidth=linewide,label='B')
    plt.plot(t1,pop2[:,2],linewidth=linewide,label='C') 
#    plt.plot(t1,pop2[:,3],linewidth=linewide,label='D') 
    plt.title('Populations evolution')
    plt.xlabel('time, ps')
    plt.ylabel('N')
    plt.xlim(0,5)
    plt.legend(bbox_to_anchor=(0., 1.05, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()   
    i=0
    
    if interm<0:
        ssignwoi1=np.empty_like(ssign1)
        for i in range(len(wavenumbers)):
            w=np.sqrt(np.diag(weights[:,i]))
            popw1=np.dot(w,pop1)
            dataw1=np.dot(data[:,i],w)
            ssign1[:,i]=linalg.lstsq(popw1,dataw1)[0]

    else:
        ssignwoi1=np.empty((len(pop0)-1,len(wavenumbers)))
        for i in range(len(wavenumbers)):
            w=np.sqrt(np.diag(weights[:,i]))
            popw1=np.dot(w,pop1)
            popwoi1=np.delete(popw1,interm,axis=1)
            dataw1=np.dot(data[:,i],w)
            ssignwoi1[:,i]=linalg.lstsq(popwoi1,dataw1)[0]
        ssign1=np.insert(ssignwoi1,interm,np.zeros(len(wavenumbers)),axis=0)
    
    i=0
    ssign2=np.empty((len(pop0),len(wavenumbers)))
    for i in range(len(ssign1[:,0])):
        ssign2[i,:]=pop1[2,i]*ssign1[i,:]

    popex=np.empty((len(ta),len(pop0)))
    for i in range(len(ta)):
        popex[i,:]=np.dot(linalg.expm(ratem*ta[i]),pop0)
    fit=np.dot(popex,ssign1)        

    # Plotting the fit result
    plt.figure(figsize=(12,6))
    timeplt=np.array([0,3,6,8,10,12,15,20,22])
    color=iter(plt.cm.rainbow(np.linspace(1,0,len(range(len(timeplt))))))
    for i in timeplt:
        c=next(color)
        plt.plot(wavenumbers,data[i,:],linestyle='None',marker='o',markersize=3,color=c)
        plt.plot(wavenumbers,fit[i,:],linewidth=linewide,color=c,label=str(t[i])+' ps') 

    plt.plot(wavenumbers,np.zeros_like(wavenumbers))
    plt.xticks(np.arange(np.min(wavenumbers).astype(int)//200+1,np.max(wavenumbers).astype(int)//200+1,1)*200)
    plt.grid()
    plt.xlabel(r'$Wavenumber, cm^{-1}$',size=14)
    plt.ylabel(r'$\Delta\alpha$',size=20)
    plt.xlim(np.min(wavenumbers),np.max(wavenumbers))
    plt.legend(bbox_to_anchor=(0., 1.05, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
    plt.show()

#    # Optional save the result
#    location=os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
#    newdir=os.path.dirname(os.path.join(location,'data and fit/'))
#    print(newdir)
#    if not os.path.exists(newdir):
#        os.makedirs(newdir)
#    filename='Data_w12CTAB.dat'
#    file=os.path.join(newdir,filename)
#    result=np.insert(np.transpose(isosignal[:,cuta:]),0,wavenumbers,axis=0)
#    time1=np.insert(ta,0,0)
#    result=np.insert(result,0,time1,axis=1)
#    np.savetxt(file,result,delimiter=' ',header='Y - time, X - wavenumbers')
#    
#    location=os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
#    newdir=os.path.dirname(os.path.join(location,'data and fit/'))
#    print(newdir)
#    if not os.path.exists(newdir):
#        os.makedirs(newdir)
#    filename='Fit_w12CTAB.dat'
#    file=os.path.join(newdir,filename)
#    result=np.insert(fit,0,wavenumbers,axis=0)
#    time1=np.insert(ta,0,0)
#    result=np.insert(result,0,time1,axis=1)
#    np.savetxt(file,result,delimiter=' ',header='Y - time, X - wavenumbers')

    return [ssign1,pop1]

[sign,pop]=signatures(p0)    

# Plot the calculated spectral signatures
plt.figure(figsize=(12,6))
plt.plot(wavenumbers,sign[0,:],linewidth=linewide,label='A')   
plt.plot(wavenumbers,sign[2,:],linewidth=linewide,label='C')   
plt.plot(wavenumbers,sign[1,:],linewidth=linewide,label='B')
plt.title('Spectral signatures')
plt.xlabel(r'$Wavenumber, cm^{-1}$',size=14)
plt.ylabel(r'$\Delta\alpha$',size=20)
plt.xlim(np.min(wavenumbers),np.max(wavenumbers))
plt.xticks(np.arange(np.min(wavenumbers).astype(int)//200+1,np.max(wavenumbers).astype(int)//200+1,1)*200)
plt.grid()
plt.legend(bbox_to_anchor=(0., 1.05, 1, .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
plt.show()    


# Time constants from the fit
print(np.reciprocal(p0))

# Saving the spectral signatures and optimized time constants
location=os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
newdir=os.path.dirname(os.path.join(location,'data and fit/'))
if not os.path.exists(newdir):
    os.makedirs(newdir)
filename='Signatures_w12_2600.dat' 
file=os.path.join(newdir,filename)
result=np.insert(sign,0,wavenumbers,axis=0)
np.savetxt(file,result,delimiter=' ',header='A-'+np.str(1/p0[0])[:5]+'->B-'+np.str(1/p0[1])[:5]+'->C-')
