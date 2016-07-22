"""A script to import a dumped pickle file from the pipeline and plot the flux as a function of zaber position to determine if the alignment was correct.




"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import glob
import pdb
import pickle
import matplotlib.cm as cm
import scipy.interpolate as interpolate

f=open('all.pkl','rb')
wave,fluxes,flat_flux,arc_flux,lenslet_ims,xpos,ypos=pickle.load(f)



#lenslet_flat = np.median(np.median(flat_flux,axis=0),axis=0)[1:].reshape(3,3)

#flatnorm=lenslet_flat/(np.median(lenslet_flat))

#lenslet_ims_f=lenslet_ims / flatnorm

height=np.sum(np.sum(lenslet_ims,2),1)

#plt.scatter(xpos,ypos,c=np.sum(np.sum(lenslet_ims_f,2),1),s=40,cmap=cm.gist_heat)
#plt.title('flat')
#plt.figure()
#plt.scatter(xpos,ypos,c=np.sum(np.sum(lenslet_ims,2),1),s=40,cmap=cm.gist_heat)
#plt.show()

#Linearly interpolate over the values of the height over 500 positions to make the image clearer to the eye.
numIndexes = 500
xi = np.linspace(np.min(xpos), np.max(xpos),numIndexes)
yi = np.linspace(np.min(ypos), np.max(ypos),numIndexes)

XI, YI = np.meshgrid(xi, yi)
points = np.vstack((xpos,ypos)).T
values = np.asarray(height)
points = np.asarray(points)
#values = np.asarray(estimatedHeightList)
DEM = interpolate.griddata(points, values, (XI,YI), method='linear')

#Now plot all.
plt.imshow(DEM,cmap ='RdYlGn_r',origin='lower',extent=[np.min(xpos), np.max(xpos),np.min(ypos), np.max(ypos)] )
plt.colorbar()
plt.scatter(xpos,ypos,c=np.sum(np.sum(lenslet_ims,2),1),s=40,cmap=cm.gist_heat,alpha=0.25)
plt.title('Global flux using all fibers')


#Now do the same considering only the flux from single fibers

#This parameter has the median of the medians of the fluxes for fibers with flux on them for each frame. 
ind_lenslet=np.zeros((fluxes.shape[0],fluxes.shape[3]))
for i in range(fluxes.shape[0]):
    ind_lenslet[i]=np.median(np.median(fluxes[i,16:21,:,:],axis=1),axis=0)


#New figure, using subplots for each fiber, hopefully the locations are roughly correct. 
plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    height=ind_lenslet[:,i]
    numIndexes = 500
    xi = np.linspace(np.min(xpos), np.max(xpos),numIndexes)
    yi = np.linspace(np.min(ypos), np.max(ypos),numIndexes)
    XI, YI = np.meshgrid(xi, yi)
    points = np.vstack((xpos,ypos)).T
    values = np.asarray(height)
    points = np.asarray(points)
    #values = np.asarray(estimatedHeightList)
    DEM = interpolate.griddata(points, values, (XI,YI), method='linear')
    plt.imshow(DEM,cmap ='RdYlGn_r',origin='lower',extent=[np.min(xpos), np.max(xpos),np.min(ypos), np.max(ypos)] )
    plt.colorbar()
    plt.title('Using only fiber'+str(i+1))

plt.show()

