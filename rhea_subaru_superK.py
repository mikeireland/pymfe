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
import astropy.modeling as amod
plt.ion()

savefile="Focus00.pkl"
dir = "/Users/mireland/data/rhea_subaru/160221/Focus00/"

savefile="Focus30.pkl"
dir = "/Users/mireland/data/rhea_subaru/160221/Focus30/"

savefile="Focus60.pkl"
dir = "/Users/mireland/data/rhea_subaru/160221/Focus60/"

savefile="1603.pkl"
dir = "/Users/mireland/data/rhea_subaru/160317/dither_final/"

savefile="1603_initial.pkl"
dir = "/Users/mireland/data/rhea_subaru/160317/dither_initial/"

star_files = glob.glob(dir + "*.fits")

nstars = len(star_files)
lenslet_ims = np.empty( (nstars,3,3) )
xpos = np.empty( (nstars) )
ypos = np.empty( (nstars) )

rhea2_format = pymfe.rhea.Format(spect='subaru',mode='slit')
rhea2_extract = pymfe.Extractor(rhea2_format, transpose_data=True)
xx, wave, blaze = rhea2_format.spectral_format()

fluxes = []
for i in range(nstars):
    star_data = pyfits.getdata(star_files[i])
    star_data -= np.median(star_data[0:500,:])

    hh = pyfits.getheader(star_files[i])
    xpos[i] = hh['ZABERX']
    ypos[i] = hh['ZABERY']    

    flux,var = rhea2_extract.one_d_extract(data=star_data.T, rnoise=20.0)
    fluxes.append(flux)

    lenslet_ims[i,:,:] = np.median(np.median(flux[12:20,:,:],axis=0),axis=0)[1:].reshape(3,3)
    lenslet_ims[i,1,:] = lenslet_ims[i,1,::-1]
    
    plt.imshow(lenslet_ims[i,:,:],interpolation='nearest', cmap=cm.gray)

pickle.dump((lenslet_ims,xpos,ypos), open(savefile, 'wb'))
plt.clf()
plt.scatter(xpos,ypos,s=100,c=np.sum(np.sum(lenslet_ims,2),1),cmap=cm.gist_heat)
