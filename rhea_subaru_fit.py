"""A script to fit tramlines etc for RHEA@Subaru data.

Long wavelengths are down and right. 15 lines visible.

lines = np.loadtxt('argon.txt')
order = 1e7/31.6*2*np.sin(np.radians(64.0))/argon
plt.plot(1375 - (order - np.round(order))/order*1.8e5)
plt.plot(1375 - (order - np.round(order)+1)/order*1.8e5)
plt.plot(1375 - (order - np.round(order)-1)/order*1.8e5)

Super-bright Neon line may be 7032.
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
import shutil
import matplotlib.cm as cm
plt.ion()


#Define the files in use (NB xmod.txt and wavemod.txt should be correct)
arc_file  = "/Users/mireland/data/rhea_subaru/images/20160216133539.fits"
flat_file = "/Users/mireland/data/rhea_subaru/images/20160216133507.fits"

arc_file  = "/Users/mireland/data/rhea_subaru/images/20160217210647.fits"
flat_file = "/Users/mireland/data/rhea_subaru/images/20160217210708.fits"

#March 2016
arc_file  = "/Users/mireland/data/rhea_subaru/"
flat_file = "/Users/mireland/data/rhea_subaru/"


rhea2_format = pymfe.rhea.Format(spect='subaru')
rhea2_extract = pymfe.Extractor(rhea2_format, transpose_data=True)
xx, wave, blaze = rhea2_format.spectral_format()

flat_data = pyfits.getdata(flat_file)
arc_data = pyfits.getdata(arc_file)

flat_data -= np.median(flat_data)
arc_data -= np.median(arc_data)

nx = arc_data.shape[0]
ny = arc_data.shape[1]

flat_flux,flat_var = rhea2_extract.one_d_extract(data=flat_data.T, rnoise=20.0)
arc_flux,arc_var = rhea2_extract.one_d_extract(data=arc_data.T, rnoise=20.0)

#First, examine the flat file and find the correct order by identifying the key arc line
#at 7032 Angstroms.
data_to_show = arc_data - 0.03*flat_data
plt.clf()
plt.imshow( np.arcsinh( (data_to_show-np.median(data_to_show))/1e2) , interpolation='nearest', aspect='auto', cmap=cm.gray)
plt.axis([0,ny,nx,0])
print("Click on 7032 Angstrom line in order 81 (super bright, with a faint line to its left)")
xy = plt.ginput(1)
#NB "X" and "Y" back to front for RHEA@Subaru.
ypix = xy[0][0]
xpix = xy[0][1]
ref_wave = 7032.4131

#Now, tweak the xmod and check that the flat really works.

#The reference line is order 82
m_ix = 81-rhea2_format.m_min
model_ref_y = np.interp(ref_wave, wave[m_ix],np.arange(ny))
model_ref_x = np.interp(model_ref_y, np.arange(ny), xx[m_ix]) + nx//2
model_ref_wave = wave[m_ix][int(ypix)]

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
xx, wave, blaze = rhea2_format.spectral_format()
plt.plot(xx.T + nx//2,'b')

print("Press <Enter> to continue... (Ctrl-C if a problem!)")
dummy = raw_input()

#Re-fit
rhea2_format.fit_x_to_image(flat_data.T)
shutil.copyfile('xmod.txt', 'data/subaru/xmod.txt')

#Now find the other lines, after first re-loading into the extractor.
rhea2_extract = pymfe.Extractor(rhea2_format, transpose_data=True)
rhea2_extract.find_lines(arc_data.T, arcfile='data/subaru/neon.txt',flat_data=flat_data.T)
#cp arclines.txt data/subaru/
shutil.copyfile('data/subaru/arclines.txt','data/subaru/arclines.backup')
shutil.copyfile('arclines.txt', 'data/subaru/arclines.txt')

#Now finally do the wavelength fit!
rhea2_format.read_lines_and_fit()
shutil.copyfile('wavemod.txt', 'data/subaru/wavemod.txt')
