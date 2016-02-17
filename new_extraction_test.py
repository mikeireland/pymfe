"""A script testing the extraction pipeline of RHEA

Steps
1) Initialise Format, Extractor and RadialVelocity
2) Define file paths for science, flat and dark frames
3) Extract spectra
4) Create/import reference spectra
5) Calculate radial velocities
6) Plot radial velocities
"""
import numpy as np
try:
    import pyfits
except:
    import astropy.io.fits as pyfits
import pymfe
import glob
from astropy.coordinates import SkyCoord
from astropy import units as u

    
#===============================================================================
# Parameters/Constants/Variables
#===============================================================================
do_bcor = True
med_cut = 0.6
plot_title = "nuOphiuchi"
coord = SkyCoord('01 44 04.08338 -15 56 14.9262',unit=(u.hourangle, u.deg))

#===============================================================================
# Observations, Flats and Darks
#===============================================================================
base_path = "D:\\TestData\\"

save_file = "gammaCrucis_save.fits"

star = "gammaCrucis"
files = glob.glob(base_path + "2015*\\*" + star + "*[0123456789].fit")

# Note: Masterdark_target.fit copied from "\20150628\spectra_paper\"
star_dark = pyfits.getdata(base_path + "Dark frames\\Masterdark_target.fit")
flat_dark = pyfits.getdata(base_path + "Dark frames\\Masterdark_flat.fit")

file_dirs = [f[f.rfind("\\")-8:f.rfind("\\")] for f in files]
flat_files = [base_path + f + "\\" + f + "_Masterflat.fit" for f in file_dirs]


#===============================================================================
# Spectra extraction, calculation of RVs and plotting
#===============================================================================
# Initialise objects
rhea2_format = pymfe.rhea.Format()
rhea2_extract = pymfe.Extractor(rhea2_format, transpose_data=False)
xx, wave, blaze = rhea2_format.spectral_format()
rv = pymfe.rv.RadialVelocity()

# Extract spectra ("wave" removed)
fluxes, vars, bcors, mjds, dates = rv.extract_spectra(files, star_dark,  
                                                     flat_files, flat_dark, 
                                                     rhea2_extract, coord=coord,
                                                     outfile=save_file, 
                                                     do_bcor=do_bcor)
                                                  
# Create reference spectrum
wave_ref, ref_spect = rv.create_ref_spect(wave, fluxes, vars, bcors, 
                                          med_cut=med_cut)

# Calculate RVs
rvs, rv_sigs = rv.calculate_rv_shift(wave_ref, ref_spect, fluxes, wave, bcors, 
                                     vars)  
                                           

# Plot RVs
rv.plot_rvs(rvs, rv_sigs, mjds, dates, bcors, plot_title, fluxes.shape[0], 
            fluxes.shape[1],
            fluxes.shape[2])              