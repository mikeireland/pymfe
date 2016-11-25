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
#plt.ion()


#Define the files in use (NB xmod.txt and wavemod.txt should be correct)
#arc_file  = "/home/jbento/code/pymfe/data/ghost/blue/std/arcstd_blue.fits"
flat_file = "/home/jbento/code/pymfe/data/ghost/red/high/flathigh_red.fits"


#instantiate the ghostsim arm
ghost_format = pymfe.ghost.Arm('red',mode='high')

#Create an initial model of the spectrograph.
xx, wave, blaze= ghost_format.spectral_format()

#Get the data and normalize by median
flat_data = pyfits.getdata(flat_file)
#arc_data = pyfits.getdata(arc_file)

nx = flat_data.shape[0]
ny = flat_data.shape[1]


x = flat_data.shape[0]
profilex = np.arange(x) - x // 2
# Now create a model of the slit profile
mod_slit = np.zeros(x)
if ghost_format.mode == 'high':
    nfibers = 26
else:
    nfibers = ghost_format.nl

for i in range(-nfibers // 2, nfibers // 2):
    mod_slit += np.exp(-(profilex - i * ghost_format.fiber_separation)**2 /
                       2.0 / ghost_format.profile_sigma**2)


plt.plot(flat_data[:,1000])
x=np.arange(flat_data[:,0].shape[0])
plt.plot(x-1,mod_slit*400)
plt.show()

#Have a look at the default model and make small adjustments if needed.
flat_conv=ghost_format.slit_flat_convolve(flat_data)
ghost_format.adjust_model(flat_conv,convolve=False,percentage_variation=10)

#Re-fit
ghost_format.fit_x_to_image(flat_conv,decrease_dim=8,inspect=True)
#shutil.copyfile('xmod.txt', 'data/subaru/xmod.txt')


'''
#Now find the other lines, after first re-loading into the extractor.
ghost_extract = pymfe.Extractor(ghost_format, transpose_data=True)
ghost_extract.find_lines(arc_data.T, arcfile='data/subaru/neon.txt',flat_data=flat_data.T)
#cp arclines.txt data/subaru/
shutil.copyfile('data/subaru/arclines.txt','data/subaru/arclines.backup')
shutil.copyfile('arclines.txt', 'data/subaru/arclines.txt')

#Now finally do the wavelength fit!
ghost_format.read_lines_and_fit()
shutil.copyfile('wavemod.txt', 'data/subaru/wavemod.txt')
'''
