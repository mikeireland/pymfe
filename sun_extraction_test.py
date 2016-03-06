"""A script testing the extraction pipeline of RHEA

Steps
1) Initialise Format, Extractor and RadialVelocity
2) Define file paths for science, flat and dark frames
3) Extract/import spectra
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
do_bcor = False
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
# Science Frames
#star = "gammaCrucis"
#star = "tauCeti"
#star = "thar"
star = "sun"
base_path = "/priv/mulga1/jbento/rhea2_data/20160221_sun/"
files = glob.glob(base_path + "*" + star + "*.FIT*")

# Flats and Darks
# Note: Masterdark_target.fit copied from "\20150628\spectra_paper\"
#star_dark = pyfits.getdata(base_path + "Dark frames\\Masterdark_target.fit")
#flat_dark = pyfits.getdata(base_path + "Dark frames\\Masterdark_flat.fit")

#file_dirs = [f[f.rfind("\\")-8:f.rfind("\\")] for f in files]
flat_files = [base_path + "20151130_Masterflat_calibrated.fit"]*len(files)

# Set to len(0) arrays when extracting ThAr
star_dark = np.empty(0)
flat_dark = np.empty(0)
#flat_files = np.empty(0)

# Extracted spectra output
out_path = "/priv/mulga1/arains/Solar_Extracted/"
extracted_files = glob.glob(out_path + "2015*" + star + 
                            "*[0123456789]_extracted.fits")

# Saved reference spectrum
ref_path = out_path + "reference_spectrum_74gammaCrucis.fits"                            
                            
# RV csv output
base_rv_path = out_path + star
print len(files), len(flat_files)
#===============================================================================
# Extract and save spectra/load previously extracted spectra
#===============================================================================
files.sort()
files = files[100:]

# Extract spectra ("wave" removed)
# OPTION 1: Extract and save spectra
fluxes, vars, bcors, mjds = rv.extract_spectra(files, rhea2_extract, 
                                               star_dark=star_dark, 
                                               flat_files=flat_files,
                                               flat_dark=flat_dark, 
                                              coord=coord, do_bcor=do_bcor)
                                                     
# Save spectra (Make sure to save "wave" generated from rhea2_format)
rv.save_fluxes(files, fluxes, vars, bcors, wave, mjds, out_path)                                                     

# OPTION 2: Load previously extracted spectra
#fluxes, vars, wave, bcors, mjds = rv.load_fluxes(extracted_files)

#===============================================================================
# Create and save/import reference spectrum
#===============================================================================                                                     
# OPTION 1: Create and save a new reference spectrum
wave_ref, ref_spect = rv.create_ref_spect(wave, fluxes[:10,:,:], vars[:10,:,:], bcors[:10], 
                                          med_cut=med_cut)

rv.save_ref_spect(files[:10], ref_spect, vars, wave_ref, bcors, mjds, out_path)                                          
                                       
# OPTION 2: Import a pre-existing reference spectrum                                          
#ref_spect, vars_ref, wave_ref, bcors_ref, mjds_ref = rv.load_ref_spect(ref_path)

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