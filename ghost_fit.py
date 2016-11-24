"""A script to fit tramlines etc for Ghost data.


"""

from __future__ import division, print_function
import pymfe
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import pdb
import shutil
import matplotlib.cm as cm
plt.ion()


#Define the files in use (NB xmod.txt and wavemod.txt should be correct)
arc_file  = "/home/jbento/arc_blue.fits"
flat_file = "/home/jbento/flat_blue.fits"


#instantiate the ghostsim arm
ghost_format = pymfe.ghost.Arm('blue',mode='std')

#Create an initial model of the spectrograph.
xx, wave, blaze= ghost_format.spectral_format()

#Get the data and normalize by median
flat_data = pyfits.getdata(flat_file)
arc_data = pyfits.getdata(arc_file)

nx = arc_data.shape[0]
ny = arc_data.shape[1]

#Have a look at the default model and make small adjustments if needed.
ghost_format.adjust_model(flat_data)



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
