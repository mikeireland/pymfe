"""A script to fit tramlines etc for RHEA@Subaru data.

Long wavelengths are down and right. 15 lines visible.

lines = np.loadtxt('argon.txt')
order = 1e7/31.6*2*np.sin(np.radians(64.0))/argon
plt.plot(1375 - (order - np.round(order))/order*1.8e5)
plt.plot(1375 - (order - np.round(order)+1)/order*1.8e5)
plt.plot(1375 - (order - np.round(order)-1)/order*1.8e5)
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
plt.ion()

dir = "/Users/mireland/data/rhea2/rhea_subaru_testdata/final_testing/"

all_files = glob.glob(dir + "*.fits")

flat_files = all_files[55:58]
arc_files = all_files[52:55]

rhea2_format = pymfe.rhea.Format(spect='subaru')
rhea2_extract = pymfe.Extractor(rhea2_format, transpose_data=True)
xx, wave, blaze = rhea2_format.spectral_format()


flat_data = pyfits.getdata(flat_files[0])
arc_data = pyfits.getdata(arc_files[0])
flat_flux,flat_var = rhea2_extract.one_d_extract(data=flat_data.T, rnoise=20.0)
arc_flux,arc_var = rhea2_extract.one_d_extract(data=arc_data.T, rnoise=20.0)

