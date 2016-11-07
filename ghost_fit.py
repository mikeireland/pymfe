"""A script to fit tramlines etc for Ghost data.


"""

from __future__ import division, print_function
import pymfe,pyghost
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
import shutil
import matplotlib.cm as cm
plt.ion()


#Define the files in use (NB xmod.txt and wavemod.txt should be correct)
arc_file  = "/home/jbento/arc_blue.fits"
flat_file = "/home/jbento/flat_blue.fits"


#instantiate the ghostsim arm
ghost_format = pymfe.ghost.Arm('blue',mode='std')
#This function adds a few things to the ARm class that are needed in the Extractor.
ghost_format.set_mode('std')

extract = pymfe.Extractor(ghost_format, transpose_data=True)
#Create an initial model of the spectrograph.
xx, wave, blaze,ccd_centre = ghost_format.spectral_format()

#Get the data and normalize by median
flat_data = pyfits.getdata(flat_file)
arc_data = pyfits.getdata(arc_file)

flat_data -= np.median(flat_data)
arc_data -= np.median(arc_data)

nx = arc_data.shape[0]
ny = arc_data.shape[1]


flat_flux,flat_var = extract.two_d_extract(data=flat_data.T, rnoise=3.0)
arc_flux,arc_var = extract.two_d_extract(data=arc_data.T, rnoise=3.0)
pdb.set_trace()
#First, examine the arc file and find the correct order by identifying the key arc line
#at 7032 Angstroms.
data_to_show = arc_data - 0.03*flat_data
plt.clf()
plt.imshow( np.arcsinh( (data_to_show-np.median(data_to_show))/1e2) , interpolation='nearest', aspect='auto', cmap=cm.gray)
plt.axis([0,ny,nx,0])
print("Click on 4519.259 Angstrom line in order 76 (bright, relatively isolated, to the left of a series of faint lines.)")
plt.plot(1708,1955)
xy = plt.ginput(1)
#NB "X" and "Y" back to front for RHEA@Subaru.
ypix = xy[0][0]
xpix = xy[0][1]
ref_wave = 4519.259

#Now, tweak the xmod and check that the flat really works.

#The reference line is order 76
m_ix = 76-ghost_format.order_min
model_ref_y = np.interp(ref_wave, wave[m_ix],np.arange(ny))
model_ref_x = np.interp(model_ref_y, np.arange(ny), xx[m_ix]) + nx//2
model_ref_wave = wave[m_ix][int(ypix)]

"""
#Based on this new data, tweak the xmod and arc files.
shutil.copyfile('data/subaru/xmod.txt','data/subaru/xmod.backup')
xmod = np.loadtxt('data/subaru/xmod.txt')
xmod[-1,-1] += xpix - model_ref_x
np.savetxt('data/subaru/xmod.txt',xmod,fmt='%.4e')

shutil.copyfile('data/subaru/wavemod.txt','data/subaru/wavemod.backup')
wavemod = np.loadtxt('data/subaru/wavemod.txt')
wavemod[-1,-1] += ref_wave - model_ref_wave
np.savetxt('data/subaru/wavemod.txt',wavemod,fmt='%.6e')


#Reload and plot...
xx, wave, blaze = ghost_format.spectral_format()
plt.plot(xx.T + nx//2,'b')

print("Press <Enter> to continue... (Ctrl-C if a problem!)")
dummy = raw_input()

#Re-fit
ghost_format.fit_x_to_image(flat_data.T)
shutil.copyfile('xmod.txt', 'data/subaru/xmod.txt')

#Now find the other lines, after first re-loading into the extractor.
ghost_extract = pymfe.Extractor(ghost_format, transpose_data=True)
ghost_extract.find_lines(arc_data.T, arcfile='data/subaru/neon.txt',flat_data=flat_data.T)
#cp arclines.txt data/subaru/
shutil.copyfile('data/subaru/arclines.txt','data/subaru/arclines.backup')
shutil.copyfile('arclines.txt', 'data/subaru/arclines.txt')

#Now finally do the wavelength fit!
ghost_format.read_lines_and_fit()
shutil.copyfile('wavemod.txt', 'data/subaru/wavemod.txt')
"""
