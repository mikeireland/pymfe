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
from astropy.time import Time
import astropy.coordinates as coordinates
from astropy.coordinates import SkyCoord
from astropy import units as u
import PyAstronomy.pyasl as pyasl
import pdb    
#===============================================================================
# Parameters/Constants/Variables/Initialisation 
#===============================================================================
# Constants/Variables
do_bcor = False
med_cut = 0.6
coord = SkyCoord('01 44 04.08338 -15 56 14.9262',unit=(u.hourangle, u.deg))

# Initialise objects
rhea2_format = pymfe.rhea.Format()
rhea2_format.fib_image_width_in_pix = 7.0 #Attempted over-write as a test
rhea2_extract = pymfe.Extractor(rhea2_format, transpose_data=False)
xx, wave, blaze = rhea2_format.spectral_format()
rv = pymfe.rv.RadialVelocity()

#Q-factor test showed that with 10 orders, we should be getting 4m/s rms per frame
# 3e8/5e3/np.sqrt(4e4*0.3*2000*10)
# c/(Q*sqrt(Ncounts*ccdgain*npix_per_order*norders)
#dd = pyfits.getdata(files[0])
#plt.imshow(dd.T, aspect='auto',cmap=cm.gray,interpolation='nearest')
#plt.plot(xx.T + dd.shape[1]/2)

#===============================================================================
# File paths (Observations, Flats and Darks, save/load directories) 
#===============================================================================
# Science Frames
#star = "thar"
star = "sun"
base_path = "/priv/mulga1/jbento/rhea2_data/20160221_sun/"
files = glob.glob(base_path + "*" + star + "*.FIT*")

# Note that the solar data has already been dark corrected
flat_files = [base_path + "20151130_Masterflat_calibrated.fit"]*len(files)
files.sort()

# Remove bad section (composed of frames 913, 914 and 915)
files.pop(912)
files.pop(912)
files.pop(912)

# Set to len(0) arrays when extracting ThAr
#star_dark = np.empty(0)
#flat_dark = np.empty(0)
#flat_files = np.empty(0)

# Extracted spectra output
out_path = "/priv/mulga1/arains/Solar_Extracted/"
extracted_files = glob.glob(out_path + "*" + star + "*extracted.fits")
extracted_files.sort()

# Again, don't consider  bad section, but this time from previously extracted
# and saved data
extracted_files.pop(912)
extracted_files.pop(912)
extracted_files.pop(912)

# Saved reference spectrum
ref_path = out_path + "reference_spectrum_10_sun.fits"                            
                            
# RV csv output
base_rv_path = out_path + star

#===============================================================================
# Extract and save spectra
#===============================================================================
# Extract spectra
#fluxes, vars, bcors, mjds = rv.extract_spectra(files, rhea2_extract, 
#                                               star_dark=star_dark, 
#                                               flat_files=flat_files,
#                                               flat_dark=flat_dark, 
#                                              coord=coord, do_bcor=do_bcor)
                                                     
# Save spectra (Make sure to save "wave" generated from rhea2_format)
#rv.save_fluxes(files, fluxes, vars, bcors, wave, mjds, out_path)                                                     

#===============================================================================
# Create and save/import reference spectrum
#===============================================================================                                                     
# OPTION 1: Create and save a new reference spectrum
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
    
    bcors = []
    
    # Calculate the barycentric correction for each observation, based on the
    # instantaneous position of the sun
    for t in mjds:
        time = Time(t, format="mjd")
        coord = SkyCoord(coordinates.get_sun(time))
        location = location=('151.2094','-33.865',100.0)
        
        bcors.append(1e3*pyasl.helcorr(float(location[0]), float(location[1]),
                     location[2], coord.ra.deg, coord.dec.deg, time.jd)[0] )
    
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
