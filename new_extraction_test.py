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
# Parameters/Constants/Variables/Initialisation 
#===============================================================================
# Constants/Variables
do_bcor = True
med_cut = 0.6
plot_title = "gammaCrucis"
coord = SkyCoord('01 44 04.08338 -15 56 14.9262',unit=(u.hourangle, u.deg))

# Initialise objects
rhea2_format = pymfe.rhea.Format()
rhea2_extract = pymfe.Extractor(rhea2_format, transpose_data=False)
xx, wave, blaze = rhea2_format.spectral_format()
rv = pymfe.rv.RadialVelocity()

#===============================================================================
# File paths (Observations, Flats and Darks, save/load directories) 
#===============================================================================
base_path = "D:\\rhea_data\\"

save_file = "gammaCrucis_save.fits"

star = "gammaCrucis"
files = glob.glob(base_path + "2015*\\*" + star + "*[0123456789].fit")

# Note: Masterdark_target.fit copied from "\20150628\spectra_paper\"
star_dark = pyfits.getdata(base_path + "Dark frames\\Masterdark_target.fit")
flat_dark = pyfits.getdata(base_path + "Dark frames\\Masterdark_flat.fit")

file_dirs = [f[f.rfind("\\")-8:f.rfind("\\")] for f in files]
flat_files = [base_path + f + "\\" + f + "_Masterflat.fit" for f in file_dirs]

base_rv_path = base_path + star

out_path = "D:\\Extracted\\"
extracted_files = glob.glob(out_path + "2015*" + star + 
                            "*[0123456789]_extracted.fits")

#===============================================================================
# Extract and save spectra/load previously extracted spectra
#===============================================================================
# Extract spectra ("wave" removed)
# OPTION 1: Extract and save spectra
#fluxes, vars, bcors, mjds = rv.extract_spectra(files, star_dark, flat_files,
#                                                     flat_dark, rhea2_extract, 
#                                                     coord=coord,
#                                                     do_bcor=do_bcor)

# Save spectra (Make sure to save "wave" generated from rhea2_format)
#rv.save_fluxes(files, fluxes, vars, bcors, wave, mjds, out_path)                                                     

# OPTION 2: Load previously extracted spectra
fluxes, vars, wave, bcors, mjds = rv.load_fluxes(extracted_files)

# Extract dimensions (Number of files, orders and pixels/order respectively)
nf = fluxes.shape[0]
nm = fluxes.shape[1]
ny = fluxes.shape[2]

#===============================================================================
# Create and save/import reference spectrum
#===============================================================================                                                     
# OPTION 1: Create and save a new reference spectrum
wave_ref, ref_spect = rv.create_ref_spect(wave, fluxes, vars, bcors, 
                                          med_cut=med_cut)

# OPTION 2: Import a pre-existing reference spectrum                                          
# wave_ref, ref_spect = some_import_function(...)

#===============================================================================
# Calculate, save and plot radial velocities
#===============================================================================                                          
# Calculate RVs
rvs, rv_sigs = rv.calculate_rv_shift(wave_ref, ref_spect, fluxes, vars, bcors, 
                                     wave)  

# Save RVs
rv.save_rvs(rvs, rv_sigs, mjds, base_rv_path)
                                     
# Plot RVs
#rv.plot_rvs(rvs, rv_sigs, mjds, dates, bcors, plot_title)              