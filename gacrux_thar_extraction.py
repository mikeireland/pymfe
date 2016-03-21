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

# Specified header parameters
xbin = 2
ybin = 1
exptime = 120

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
star = "thar"
base_path = "/priv/mulga1/jbento/rhea2_data/gammaCrucis/"

# Find all Gacrux ThAr files and sort by observation date in MJD
all_files = np.array(glob.glob(base_path + "2015*/*" + star + "_*.fit*"))
sorted = np.argsort([pyfits.getheader(e)['JD'] for e in all_files])
all_files = all_files[sorted]
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
dark_path = base_path + "Dark frames/Masterdark_thar.fit"
star_dark = pyfits.getdata(dark_path)
flat_dark = pyfits.getdata(base_path + "Dark frames/Masterdark_flat.fit")

# Note: this particular flat was chosen as it has an exposure time of 3 seconds,
# the same length as the flat dark that will be used to correct it
flat_path = base_path + "20150527/20150527_Masterflat.fit"
flat_files = [flat_path]*len(files)

# Extracted spectra output
out_path = "/priv/mulga1/arains/Gacrux_Extracted_ThAr/"
#extracted_files = np.array(glob.glob(out_path + "*" + star + "*extracted.fits"))

# Sort to account for files not being labelled with MJD
#sorted = np.argsort([pyfits.getheader(e)['JD'] for e in extracted_files])
#extracted_files = extracted_files[sorted]
                 
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
                                               do_bcor=do_bcor)
                                                     
# Save spectra (Make sure to save "wave" generated from rhea2_format)
rv.save_fluxes(files, fluxes, vars, bcors, wave, mjds, out_path)                                                     

# OPTION 2: Load previously extracted spectra
#fluxes, vars, wave, bcors, mjds = rv.load_fluxes(extracted_files)

extracted_files = np.array(glob.glob(out_path + "*" + star + "*extracted.fits"))

# Sort to account for files not being labelled with MJD
sorted = np.argsort([pyfits.getheader(e)['JD'] for e in extracted_files])
extracted_files = extracted_files[sorted]

#===============================================================================
# Create and save/import reference spectrum
#===============================================================================                                                     
# Number of frames to use for reference spectrum
# Load the first 10 observations to use as a reference

fluxes, vars, wave, bcors, mjds = rv.load_fluxes(extracted_files[:10])
 
wave_ref, ref_spect = rv.create_ref_spect(wave, fluxes, vars, bcors, 
                                          med_cut=med_cut)

rv.save_ref_spect(extracted_files[:10], ref_spect, vars, wave_ref, bcors, mjds, 
                  out_path, star)                                        
                                       
# OPTION 2: Import a pre-existing reference spectrum                                          
#ref_spect, vars_ref, wave_ref, bcors_ref, mjds_ref = rv.load_ref_spect(ref_path)

#===============================================================================
# Barycentrically correct based on the sun's location from moment to moment
#=============================================================================== 
# This loop is messy and there is probably a nicer way to do this...but it works
# The Linux servers are not happy with opening much more than 100 files,
# crashing and displaying a too many files warning. This is despite each .fits
# file being closed when the data have been loaded from it. A similar issue does
# not occur when initially extracting the files (975 were extracted in one go
# with no issues). 

# Parameters to process files in batches of "increment"
num_files = len(extracted_files)
num_rvs_extracted = 0
increment = 100
low = 0
high = increment
all_rvs_calculated = False

# Will be concatenated at end to give final arrays
rv_list = []
rv_sig_list = []
bcors_list = [] 
mjds_list = []

# Obviously cannot open more files than exist
if high > num_files:
    high = num_files

while not all_rvs_calculated:
    num_rvs_extracted += high - low
    # Load in a segment of files
    fluxes, vars, wave, bcors, mjds = rv.load_fluxes(extracted_files[low:high])
    
    nf = fluxes.shape[0]
    nm = fluxes.shape[1]
    ny = fluxes.shape[2]

    # Calculate the RVs
    rvs, rv_sigs = rv.calculate_rv_shift(wave_ref, ref_spect, fluxes, vars,
                                         bcors, wave)  
    
    rv_list.append(rvs)
    rv_sig_list.append(rv_sigs)
    bcors_list.append(bcors)
    mjds_list.append(mjds)
    
    # Move to next segment
    low += increment
    high += increment
    
    if high > num_files:
        high = num_files   
        
    if num_rvs_extracted == num_files:
        all_rvs_calculated = True

# Done, join together and save
all_rvs = np.concatenate(rv_list)
all_rv_sigs = np.concatenate(rv_sig_list)
all_bcors = np.concatenate(bcors_list)
all_mjds = np.concatenate(mjds_list)
    
#===============================================================================
# Save the extracted radial velocities
#===============================================================================                                          
# Save RVs
bcor_rvs = all_rvs + all_bcors.repeat(nm).reshape( (num_files,nm) )  

rv.save_rvs(all_rvs, all_rv_sigs, all_bcors, all_mjds, bcor_rvs, base_rv_path)
