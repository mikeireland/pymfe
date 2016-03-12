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
do_bcor = True
med_cut = 0.6
coord = SkyCoord('01 44 04.08338 -15 56 14.9262',unit=(u.hourangle, u.deg))

# Specified header parameters
xbin = 2
ybin = 1
exptime = 1800

badpixel_mask= pyfits.getdata('/priv/mulga1/jbento/rhea2_data/badpix.fits')
badpix=np.where(badpixel_mask==1)

# Initialise objects
rhea2_format = pymfe.rhea.Format()
rhea2_extract = pymfe.Extractor(rhea2_format, transpose_data=False, 
                                badpixmask=badpix)
xx, wave, blaze = rhea2_format.spectral_format()
rv = pymfe.rv.RadialVelocity()

#===============================================================================
# File paths (Observations, Flats and Darks, save/load directories) 
#===============================================================================
# Science Frames
star = "gammaCrucis"
base_path = "/priv/mulga1/jbento/rhea2_data/gammaCrucis/"

all_files = glob.glob(base_path + "2015*/*" + star + "*.fit*")
all_files.sort()
files = []

# Only consider files that have the same exposure time and correct binning
for f in all_files:
    fits = pyfits.open(f)
    header = fits[0].header
    
    x_head = header["XBINNING"]
    y_head = header["YBINNING"]
    exp_head = header["EXPTIME"]
    
    if x_head == xbin and y_head == ybin and exp_head == exptime:
        files.append(f)
    
    fits.close()
        
# Flats and Darks
dark_path = base_path + "Dark frames/Masterdark_target_" + str(exptime) +".fits"
star_dark = pyfits.getdata(dark_path)
flat_dark = pyfits.getdata(base_path + "Dark frames/Masterdark_flat.fit")

# Note: this particular flat was chosen as it has an exposure time of 3 seconds,
# the same length as the flat dark that will be used to correct it
flat_path = base_path + "20150527/20150527_Masterflat.fit"
flat_files = [flat_path]*len(files)

# Set to len(0) arrays when extracting ThAr
star_dark = np.empty(0)
flat_dark = np.empty(0)
flat_files = np.empty(0)

# Extracted spectra output
out_path = "/priv/mulga1/arains/Gacrux_extr_no_dark/"
extracted_files = glob.glob(out_path + "*" + star + "*.fits")
extracted_files = []
                 
# RV csv output
base_rv_path = out_path + star

#===============================================================================
# Extract and save spectra/load previously extracted spectra
#===============================================================================
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
# Number of frames to use for reference spectrum
n = 10
 
# OPTION 1: Create and save a new reference spectrum
wave_ref, ref_spect = rv.create_ref_spect(wave, fluxes[:n,:,:], vars[:n,:,:],  
                                          bcors[:n], med_cut=med_cut)

rv.save_ref_spect(files[:n], ref_spect, vars[:n,:,:], wave_ref, bcors[:n], 
                  mjds[:n], out_path, star)                                          
                                       
# OPTION 2: Import a pre-existing reference spectrum                                          
#ref_spect, vars_ref, wave_ref, bcors_ref, mjds_ref = rv.load_ref_spect(ref_path)

#===============================================================================
# Calculate, save and plot radial velocities
#===============================================================================
# Calculate RVs
rvs, rv_sigs = rv.calculate_rv_shift(wave_ref, ref_spect, fluxes, vars, bcors, 
                                     wave)  

nf = fluxes.shape[0]
nm = fluxes.shape[1]
                                     
bcor_rvs = rvs + bcors.repeat(nm).reshape( (nf, nm) )                                     
                                     
# Save RVs
rv.save_rvs(rvs, rv_sigs, bcors, mjds, bcor_rvs, base_rv_path)