"""A script to fit tramlines etc for RHEA@Subaru data.

Long wavelengths are down and right. 15 lines visible.

lines = np.loadtxt('argon.txt')
order = 1e7/31.6*2*np.sin(np.radians(64.0))/argon
plt.plot(1375 - (order - np.round(order))/order*1.8e5)
plt.plot(1375 - (order - np.round(order)+1)/order*1.8e5)
plt.plot(1375 - (order - np.round(order)-1)/order*1.8e5)

Super-bright Neon line may be 7032.

15000 counts in 20s
"""

from __future__ import division, print_function
import pymfe
try:
    import pyfits
except:
    import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import glob
import opticstools as ot
import pdb
import scipy.optimize as op
import scipy.interpolate as interp
import time
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
import PyAstronomy.pyasl as pyasl
from astropy import constants as const
import matplotlib.cm as cm
import pickle
plt.ion()

dir = "/priv/mulga1/mireland/rhea_subaru/160319/"

flat_file = dir + "cal/20160319051706.fits"
arc_file = dir + "cal/20160319051726.fits"
bias_file = dir + "cal/20160319165134.fits"

dir = "/priv/mulga1/mireland/rhea_subaru/brightest_files/"

star_files = glob.glob(dir + "*.fits")
star_files.sort()

savefile = "all.pkl"
nstars = len(star_files)
lenslet_ims = np.empty( (nstars,3,3) )
xpos = np.empty( (nstars) )
ypos = np.empty( (nstars) )

rhea2_format = pymfe.rhea.Format(spect='subaru',mode='slit')
rhea2_extract = pymfe.Extractor(rhea2_format, transpose_data=True)
xx, wave, blaze = rhea2_format.spectral_format()


flat_data = pyfits.getdata(flat_file)
arc_data = pyfits.getdata(arc_file)
bias_data = pyfits.getdata(bias_file)

flat_data -= bias_data
arc_data -= bias_data

flat_flux,flat_var = rhea2_extract.one_d_extract(data=flat_data.T, rnoise=20.0)
arc_flux,arc_var = rhea2_extract.one_d_extract(data=arc_data.T, rnoise=20.0)
fluxes = []

for i in range(nstars):
    star_data = pyfits.getdata(star_files[i])
    hh = pyfits.getheader(star_files[i])
    xpos[i] = hh['ZABERX']
    ypos[i] = hh['ZABERY']    

    star_data -= bias_data
    
    flux,var = rhea2_extract.one_d_extract(data=star_data.T, rnoise=20.0)
    fluxes.append(flux)

    lenslet_ims[i,:,:] = np.median(np.median(flux,axis=0),axis=0)[1:].reshape(3,3)
    lenslet_ims[i,1,:] = lenslet_ims[i,1,::-1]
    
    plt.imshow(lenslet_ims[i,:,:],interpolation='nearest', cmap=cm.gray)

plt.clf()
plt.scatter(xpos,ypos,c=np.sum(np.sum(lenslet_ims,2),1),cmap=cm.gist_heat)
fluxes = np.array(fluxes)
pickle.dump((wave,fluxes,flat_flux,arc_flux,lenslet_ims,xpos,ypos), open(savefile, 'wb'))

#fluxes_norm = np.empty(fluxes.shape)
#for i in range(50): fluxes_norm[i] = (fluxes[i]+1.5e2)/flat_flux

#plt.plot(wave.T,np.sum(np.sum(fluxes_norm,axis=0),axis=2).T/12)
#plt.axis([7590,7690,0,1.2])
