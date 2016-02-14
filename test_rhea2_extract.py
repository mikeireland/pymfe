"""A script to test the extraction of a bunch of RHEA2 spectra.

The functions within this module should go in the Extractor if they
are general. Spectrograph specific functions should go in the RHEA module


TODO:
0) Make sure that the Th/Ar reference is created from the same epoch that the wavelength 
scale solution is made at. i.e. add a new wavelength solution script, e.g. with 
creation of new data/orderXXX.txt files from an averaged Th/Ar for each night. This
would be an extraction then a fitting of Gaussians to each line.

1) Output the reference spectrum separately, so it can be imported. This is 
*super* important because one test we want to do is to input the sun as a reference
for Tau Ceti (part of ardata.fits.gz)

2) Put extraction in a script where tramlines are tweaked using fit_x_to_image.

3) Add flat field creation scripts to this.

4) Correct for Telluric lines... (in data/ardata.fits.gz).
For Telluric lines, the wavelength scale has to be corrected epoch to epoch. 

5) Find and correct for common bad pixels.

6) The GHOST in orders 28 to 30 should be marked as high variance. 

7) Actually use a (neatened version of) this script for the gamma Crucis and
sun data.

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

dir = "/Users/mireland/data/rhea2/20150601/"

#First few thar frames...
star = "thar"
files = glob.glob(dir + "*" + star + "*00[1234].fit")

#thar frames separated by 10
star = "thar"
files = glob.glob(dir + "*" + star + "*0[012]1.fit")

#Gamma cru
star = "gacrux"
star = "thar"

files = glob.glob(dir + "*" + star + "*00[1234].fit")
files = glob.glob("/Users/mireland/data/rhea2/2015060*/*" + star + "*00[1234].fit")
#dark = pyfits.getdata(dir + "Masterdark_target.fit")

#This is "Gamma Cru"
coord = SkyCoord('12 31 09.9596 -57 06 47.568',unit=(u.hourangle, u.deg))

save_file = "gacrux06.fit"

save_file = "thar06.fit"


#nu Oph, "Sinistra". Has bad pixels. 
#star = "sinistra"
#files = glob.glob(dir + "*" + star + "*00[12345678].fit")

#save_file = "sinistra0601.fit"
#ref_file = "" #A reference spectrum file should be possible.

#This is "Sinistra"
#coord = SkyCoord('17 59 01.59191 -09 46 25.07',unit=(u.hourangle, u.deg))

#Select a dark here
dir = "/Users/mireland/data/rhea2/tauCeti/"


star = "thar"
save_file_prefix = "tauCeti_thar1114" 
star_dark = pyfits.getdata(dir + "Masterdark_thar.fit")

star = "tauCeti"
save_file_prefix = "tauCeti1114"
star_dark = pyfits.getdata(dir + "Masterdark_target.fit")

files = glob.glob("/Users/mireland/data/rhea2/tauCeti/201511*/*" + star + "*.fit")
coord = SkyCoord('01 44 04.08338 -15 56 14.9262',unit=(u.hourangle, u.deg))

flat_dark = pyfits.getdata(dir + "Masterdark_flat.fit")

rhea2_format = pymfe.rhea.Format()
rhea2_extract = pymfe.Extractor(rhea2_format, transpose_data=False)
xx, wave, blaze = rhea2_format.spectral_format()

#Things to change each time if you want. Below for star
do_we_extract=False
do_bcor=True
med_cut=0.6 #0 for Th/Ar

#Here for Th/Ar
#do_we_extract=True
#do_bcor=False
#med_cut=0

save_file = save_file_prefix + ".fits"
rv_file = save_file_prefix + "_rv.csv"
rv_sig_file = save_file_prefix + "_rv_sig.csv"

file_dirs = [f[f.rfind('/')-8:f.rfind('/')] for f in files]
flat_files = ["/Users/mireland/data/rhea2/tauCeti/" + f + "/Masterflat.fit" for f in file_dirs]
#-----------------------------------------
def rv_shift_resid(params, wave, spect, spect_sdev, spline_ref, return_spect=False):
    """Find the residuals to a fit of a (subsampled)reference spectrum to an 
    observed spectrum. 
    
    The function for parameters p[0] through p[3] is:
    
    y(x) = Ref[ wave(x) * (1 - p[0]/c) ] * exp(p[1] * x^2 + p[2] * x + p[3])
    
    Here "Ref" is a function f(wave)
    
    TODO: replace with e.g. op.minimize_scalar to account for bad pixels
    
    Parameters
    ----------
    params: 
        ...
    wave: float array
        Wavelengths for the observed spectrum.
    spect: float array
        The observed spectrum
    spect_sdev: 
        ...
    spline_ref: 
        ...
    return_spect: boolean
        Whether to return the fitted spectrum or the 
        
    wave_ref: float array
        The wavelengths of the reference spectrum
    ref: float array
        The reference spectrum
    
    Returns
    -------
    resid:
        The fit residuals
    """
    ny = len(spect)
    xx = np.arange(ny)-ny//2
    norm = np.exp(params[1]*xx**2 + params[2]*xx + params[3])
    #Lets get this sign correct. A redshift (positive velocity) means that
    #a given wavelength for the reference corresponds to a longer wavelength for the target, 
    #which in turn means that the target wavelength has to be interpolated onto shorter
    #wavelengths for the reference.
    fitted_spect = spline_ref(wave*(1.0 - params[0]/const.c.si.value))*norm
    
    if return_spect:
        return fitted_spect
    else:
        return (fitted_spect - spect)/spect_sdev

def rv_shift_jac(params, wave, spect, spect_sdev, spline_ref):
    """Jacobian function for the above. Dodgy... sure, but
    without this there seems to be numerical derivative instability.
    
    Parameters
    ----------
    params: 
        ...
    wave: float array
        Wavelengths for the observed spectrum.
    spect: float array
        The observed spectrum
    spect_sdev: 
        ...
    spline_ref: 
        ...
        
    Returns
    -------
    jac: 
        ...
    """
    ny = len(spect)
    xx = np.arange(ny)-ny//2
    norm = np.exp(params[1]*xx**2 + params[2]*xx + params[3])
    fitted_spect = spline_ref(wave*(1.0 - params[0]/const.c.si.value))*norm
    jac = np.empty( (ny,4) )
    jac[:,3] = fitted_spect/spect_sdev
    jac[:,2] = fitted_spect*xx/spect_sdev
    jac[:,1] = fitted_spect*xx**2/spect_sdev
    jac[:,0] = (spline_ref(wave*(1.0 - (params[0] + 1.0)/const.c.si.value))*norm - fitted_spect)/spect_sdev
    return jac

def create_ref_spect(wave, fluxes, vars, bcors, rebin_fact=2, gauss_sdev = 1.0, med_cut=0.6,gauss_hw=7):
    """Create a reference spectrum from a series of target spectra, after 
    correcting the spectra barycentrically. 
    
    Parameters
    ----------
    wave:
        ...
    fluxes:
        ...
    vars:
        ...
    bvors:
        ...
    rebin_fact:
        ...
    gauss_sdev:
        ...
    med_cut:
        ...
    gauss_hw:
        ...
    
    Returns
    -------
    wave_ref:
        ...
    ref_spect:
        ...
    """
    nm = fluxes.shape[1]
    ny = fluxes.shape[2]
    nf = fluxes.shape[0]

    #Create arrays for our outputs.
    wave_ref = np.empty( (nm,rebin_fact*ny + 2) )
    ref_spect = np.empty( (nm,rebin_fact*ny + 2) )

    #First, rebin everything.
    new_shape = (fluxes.shape[1],rebin_fact*fluxes.shape[2])
    fluxes_rebin = np.empty( (fluxes.shape[0],fluxes.shape[1],rebin_fact*fluxes.shape[2]) )
    for i in range(nf):
        fluxes_rebin[i] = ot.utils.regrid_fft(fluxes[i],new_shape)

    #Create the final wavelength grid.    
    for j in range(nm):
        wave_ref[j,1:-1] = np.interp(np.arange(rebin_fact*ny)/rebin_fact,np.arange(ny),wave[j,:])
        #Fill in the end wavelengths, including +/-100 km/s from the ends.
        wave_ref[j,-2] = wave_ref[j,-3] + (wave_ref[j,-3]-wave_ref[j,-4])
        wave_ref[j,0]  = wave_ref[j,1] * (const.c.si.value + 1e5)/const.c.si.value
        wave_ref[j,-1] = wave_ref[j,-2] * (const.c.si.value - 1e5)/const.c.si.value

    #Barycentric correct
    for i in range(nf):
        for j in range(nm):
            #Awkwardly, we've extended the wavelength scale by 2 elements, but haven't yet extended 
            #the fluxes...
            ww = wave_ref[j,1:-1]
            fluxes_rebin[i,j] = np.interp(ww*(1 - bcors[i]/const.c.si.value),ww[::-1],fluxes_rebin[i,j,::-1])
            
    #Subsample a reference spectrum using opticstools.utils.regrid_fft
    #and interpolate to fit. 
    flux_meds = np.median(fluxes_rebin,axis=2)
    flux_files = np.median(flux_meds,axis=1)
    if med_cut > 0:
        good_files = np.where(flux_files > med_cut*np.median(flux_files))[0]
    else:
        good_files = np.arange(len(flux_files),dtype=np.int)
    flux_orders = np.median(flux_meds[good_files],axis=0)    
    flux_norm = fluxes_rebin.copy()
    for g in good_files:
        for j in range(nm):
            flux_norm[g,j,:] /= flux_meds[g,j]
    #Create a median over files
    flux_ref = np.median(flux_norm[good_files],axis=0)
    #Multiply this by the median for each order  
    for j in range(nm):
        flux_ref[j] *= flux_orders[j]
        
    #Create a Gaussian smoothing function for the reference spectrum. This is needed to
    #prevent a bias to zero radial velocity, especially in the case of few data points.
    gg = np.exp(-(np.arange(2*gauss_hw+1)-gauss_hw)**2/2.0/gauss_sdev**2)
    gg /= np.sum(gg)
    one_order = np.empty(flux_ref.shape[1] + 2*gauss_hw)
    for j in range(nm):
        one_order[gauss_hw:-gauss_hw] = flux_ref[j,:]
        one_order[:gauss_hw] = one_order[gauss_hw]
        one_order[-gauss_hw:] = one_order[-gauss_hw-1]
        ref_spect[j,:] = np.convolve(one_order, gg, mode='same')[gauss_hw-1:1-gauss_hw]
    
    return wave_ref, ref_spect

def extract_spectra(files, star_dark, flat_files, flat_dark, location=('151.2094','-33.865',100.0), coord=None, outfile=None, do_bcor=True):
    """Extract the spectrum from a file, given a dark file, a flat file and
    a dark for the flat. 
    
    Parameters
    ----------
    files: list of strings
        One string for each file. CAn be on separate nights - a full pathname should be given.
    star_dark:
        
    flat_files: list of strings.
        One string for each star file. CAn be on separate nights - a full pathname should be given.
    flat_dark:
        
    location: (lattitude:string, longitude:string, elevation:string)
        The location on Earth where the data were taken.
    coord:
    
    outfile:
    
    do_bcor: boolean
    
    
    Returns
    -------
    fluxes:
    
    vars:
    
    wave:
    
    bcors:
    
    mjds:
    
    """
    # Initialise list of return values 
    # Each index represents a single observation
    fluxes = []
    vars = []
    dates = []
    bcors = []

    #!!! This is dodgy, as files and flat_files should go together in a dict. !!!
    for ix,file in enumerate(files):
        # Dark correct the science and flat frames
        data = pyfits.getdata(file) - star_dark
        flat = pyfits.getdata(flat_files[ix]) - flat_dark
        
        header = pyfits.getheader(file)
        
        date = Time(header['DATE-OBS'], location=location)
        dates.append(date)
        
        # Determine the barycentric correction
        if do_bcor:
            if not coord:
                coord=SkyCoord( ra=float(header['RA']) , dec=float(header['DEC']) , unit='deg')
            if not location:
                location=( float(header['LONG']), float(header['LAT']), float(header['HEIGHT']))
            #(obs_long, obs_lat, obs_alt, ra2000, dec2000, jd, debug=False)
            bcors.append( 1e3*pyasl.helcorr(float(location[0]),float(location[1]),location[2],coord.ra.deg, coord.dec.deg,date.jd)[0] )
        else:
            bcors.append(0.0)
        
        # Extract the fluxes and variance for the science and flat frames
        flux, var = rhea2_extract.one_d_extract(data=data, rnoise=20.0)
        flat_flux, fvar = rhea2_extract.one_d_extract(data=flat, rnoise=20.0)
        
        for j in range(flat_flux.shape[0]):
            medf = np.median(flat_flux[j])
            flat_flux[j] /= medf
            fvar[j] /= medf**2
        
        #Calculate the variance after dividing by the flat
        var = var/flat_flux**2 + fvar * flux**2/flat_flux**4
        
        #Now normalise the flux.
        flux /= flat_flux

        #pdb.set_trace()
        fluxes.append(flux[:,:,0])
        vars.append(var[:,:,0])

    fluxes = np.array(fluxes)
    vars = np.array(vars)
    bcors = np.array(bcors)
    mjds = np.array([d.mjd for d in dates])
    
    # Output and save the results
    if not outfile is None:
        hl = pyfits.HDUList()
        hl.append(pyfits.ImageHDU(fluxes,header))
        hl.append(pyfits.ImageHDU(vars))
        hl.append(pyfits.ImageHDU(wave))
        col1 = pyfits.Column(name='bcor', format='D', array=bcors)
        col2 = pyfits.Column(name='mjd', format='D', array=mjds)
        cols = pyfits.ColDefs([col1, col2])
        hl.append(pyfits.new_table(cols))
        hl.writeto(outfile, clobber=True)

    return fluxes,vars,wave,bcors,mjds


#------ Standard analysis --------

#Extract all data.
if do_we_extract:
    fluxes,vars,wave,bcors,mjds = extract_spectra(files, star_dark, flat_files, flat_dark, coord=coord,outfile=save_file, do_bcor=do_bcor)
if not save_file is None:
    hl = pyfits.open(save_file)
    fluxes = hl[0].data
    vars = hl[1].data
    wave = hl[2].data
    bcors = hl[3].data['bcor']
    mjds = hl[3].data['mjd']

#Create a reference spectrum if not given
wave_ref,ref_spect = create_ref_spect(wave,fluxes,vars,bcors,med_cut=med_cut)

nm = fluxes.shape[1]
ny = fluxes.shape[2]
nf = fluxes.shape[0]


rvs = np.zeros( (nf,nm) )
rv_sigs = np.zeros( (nf,nm) )
initp = np.zeros(4)
initp[0]=0.0
spect_sdev = np.sqrt(vars)
fitted_spects = np.empty(fluxes.shape)
for i in range(nf):
# !!! Uncomment here !!!
    initp[0] = -bcors[i] #Start with an initial guess that there is no intrinsic RV for the target.
    for j in range(nm):
        #This is the *only* non-linear interpolation function that doesn't take forever
        spline_ref = interp.InterpolatedUnivariateSpline(wave_ref[j,::-1], ref_spect[j,::-1])
        args = (wave[j,:],fluxes[i,j,:],spect_sdev[i,j,:],spline_ref)
        #Remove edge effects in a slightly dodgy way. 20 pixels is about 30km/s. 
        args[2][:20] = np.inf
        args[2][-20:] = np.inf
        the_fit = op.leastsq(rv_shift_resid,initp,args=args, diag=[1e3,1e-6,1e-3,1],Dfun=rv_shift_jac,full_output=True)
        #Remove bad points...
        resid = rv_shift_resid( the_fit[0], *args)
        wbad = np.where( np.abs(resid) > 7)[0]
        args[2][wbad] = np.inf
        the_fit = op.leastsq(rv_shift_resid,initp,args=args, diag=[1e3,1e-7,1e-3,1],Dfun=rv_shift_jac, full_output=True)
        #Some outputs for testing
        fitted_spects[i,j] = rv_shift_resid(the_fit[0],*args,return_spect=True)
        #Save the fit and the uncertainty.
        rvs[i,j] = the_fit[0][0]
        try:
            rv_sigs[i,j] = np.sqrt(the_fit[1][0,0])
        except:
            rv_sigs[i,j] = np.NaN
    print("Done file {0:d}".format(i))

#Plot the Barycentric corrected RVs. Note that a median over all orders is
#only a first step - a weighted mean is needed.
plt.clf()

rvs += bcors.repeat(nm).reshape( (nf,nm) )

rv_mn,wt_sum = np.average(rvs,axis=1,weights=1.0/rv_sigs**2,returned=True)
rv_mn_sig = 1.0/np.sqrt(wt_sum)
rv_med1 = np.median(rvs,1)
rv_med2 = np.median(rvs[:,3:20],1)

#plt.plot_date([dates[i].plot_date for i in range(len(dates))], rv_mn)

#plt.errorbar(mjds, rv_mn, yerr=rv_mn_sig,fmt='o')
plt.errorbar(mjds, rv_med2, yerr=rv_mn_sig,fmt='o')
plt.xlabel('Date (MJD)')
plt.ylabel('Barycentric RV (m/s)')
plt.title(star)

#Write a csv file for the RVs and the RV_sigs
np.savetxt(rv_file, np.append(mjds.reshape(nf,1),rvs,axis=1), fmt="%10.4f" + nm*", %6.1f",header="Radial velocities in m/s for each order, for each MJD epoch")
np.savetxt(rv_sig_file, np.append(mjds.reshape(nf,1),rv_sigs,axis=1), fmt="%10.4f" + nm*", %6.1f",header="Radial velocity uncertainties in m/s for each order, for each MJD epoch")

#A line for checking the image...
#dd = pyfits.getdata (BLAH)(
#plt.imshow(np.arcsinh(dd/100), aspect='auto', interpolation='nearest', cmap=cm.cubehelix)
#plt.plot(1375/2 + xx.T,np.repeat(np.arange(2200),34).reshape(2200,34))
