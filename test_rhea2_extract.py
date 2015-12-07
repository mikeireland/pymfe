"""A script to test the extraction of a bunch of RHEA2 spectra

TODO:
1) Put extraction in a script where tramlines are tweaked.
2) REVERSE the wavelength scale to correct it!!!
3) Try a zero-mean cross correlation to better match the barycentric correction
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
plt.ion()


dir = "/Users/mireland/data/rhea2/20150601/"

#First few thar frames...
star = "thar"
files = glob.glob(dir + "*" + star + "*00[1234].fit")
dark = pyfits.getdata(dir + "Masterdark_thar.fit")

#thar frames separated by 10
star = "thar"
files = glob.glob(dir + "*" + star + "*0[012]1.fit")
dark = pyfits.getdata(dir + "Masterdark_thar.fit")

#Gamma cru
star = "gacrux"
files = glob.glob(dir + "*" + star + "*00[1234].fit")
dark = pyfits.getdata(dir + "Masterdark_target.fit")

#nu Oph, "Sinistra". Has bad pixels. 
star = "sinistra"
files = glob.glob(dir + "*" + star + "*00[12345678].fit")
dark = pyfits.getdata(dir + "Masterdark_target.fit")


ref_file = "" #A reference spectrum file should be possible.
rebin_fact=2
smoothit = 3

#This is "Sinistra"
coord = SkyCoord('17 59 01.59191 -09 46 25.07',unit=(u.hourangle, u.deg))

#This is "Gamma Cru"
#coord = SkyCoord('12 31 09.9596 -57 06 47.568',unit=(u.hourangle, u.deg))

#-----------------------------------------
def rv_shift_resid(params, wave,spect,spect_sdev,spline_ref):
    """Find the residuals to a fit of a (subsampled) 
    reference spectrum to an observed spectrum. 
    
    TODO: replace with e.g. op.minimize_scalar to account for bad pixels
    
    Parameters
    ----------
    wave: float array
        Wavelengths for the observed spectrum.
    spect: float array
        The observed spectrum
    wave_ref: float array
        The wavelengths of the reference spectrum
    ref: float array
        The reference spectrum
    
    Returns
    -------
    resid:
        The fit residuals
    """
    c = 2.998e8 #Speed of light in m/s
    ny = len(spect)
    xx = np.arange(ny)-ny//2
    norm = np.exp(params[1]*xx**2 + params[2]*xx + params[3])
    fitted_spect = spline_ref(wave*(1.0 + params[0]/c))*norm
    return (fitted_spect - spect)/spect_sdev

def rv_shift_jac(params, wave,spect,spect_sdev,spline_ref):
    """Jacobian function for the above. Dodgy... sure, but
    without this there seems to be numerical derivative instability.
    """
    c = 2.998e8 #Speed of light in m/s
    ny = len(spect)
    xx = np.arange(ny)-ny//2
    norm = np.exp(params[1]*xx**2 + params[2]*xx + params[3])
    fitted_spect = spline_ref(wave*(1.0 + params[0]/c))*norm
    jac = np.empty( (ny,4) )
    jac[:,3] = fitted_spect/spect_sdev
    jac[:,2] = fitted_spect*xx/spect_sdev
    jac[:,1] = fitted_spect*xx**2/spect_sdev
    jac[:,0] = (spline_ref(wave*(1.0 + (params[0] + 1.0)/c))*norm - fitted_spect)/spect_sdev
    return jac


rhea2_format = pymfe.rhea.Format()
rhea2_extract = pymfe.Extractor(rhea2_format, transpose_data=False)
xx, wave, blaze = rhea2_format.spectral_format()

fluxes = []
vars = []
dates = []
bcors = []

for file in files:
    data = pyfits.getdata(file) - dark
    date = Time(pyfits.getheader(file)['DATE-OBS'], location=('151.2094d','-33.865d'))
    #(obs_long, obs_lat, obs_alt, ra2000, dec2000, jd, debug=False)
    bcors.append(pyasl.helcorr(151.2094, -33.865, 100.,coord.ra.deg, coord.dec.deg,date.jd))
    dates.append(date)
    flux,var = rhea2_extract.one_d_extract(data=data, rnoise=20.0)
    #pdb.set_trace()
    fluxes.append(flux[:,:,0])
    vars.append(var[:,:,0])
#    plt.imshow(data, vmin=0,vmax=1e3,aspect='auto')
 
fluxes = np.array(fluxes)
vars = np.array(vars)   
bcors = np.array(bcors)
dates = np.array(dates)
nm = fluxes.shape[1]
ny = fluxes.shape[2]
nf = len(files)


#Create a Gaussian smoothing function for the reference spectrum. This is needed to
#
gg = np.exp(-(np.arange(21)-10)**2/2.0/2.5**2)
gg /= np.sum(gg)

#If not given, we need to subsample a reference spectrum using opticstools.utils.regrid_fft
#and interpolate to fit. 
if len(ref_file)==0:
    flux_meds = np.median(fluxes,axis=2)
    flux_norm = fluxes.copy()
    for i in range(nf):
        for j in range(nm):
            flux_norm[i,j,:] /= flux_meds[i,j]
    flux_ref = np.median(flux_norm,axis=0)
    flux_ref *= np.median(flux_meds)
    new_shape = (flux.shape[0],rebin_fact*flux.shape[1])
    #!!! the multiplication by rebin_fact on the next line is actually an opticstools error !!!
    ref_spect0 = ot.utils.regrid_fft(flux_ref,new_shape)*rebin_fact
    for j in range(nm):
        ref_spect0[j,:] = np.convolve(ref_spect0[j,:], gg, mode='same')
    ref_spect = np.empty( (ref_spect0.shape[0],ref_spect0.shape[1]+2) )
    ref_spect[:,1:-1] = ref_spect0
    ref_spect[:,0] = ref_spect[:,1]
    ref_spect[:,-1] = ref_spect[:,-2]
    
#    ref_spect = np.roll(ref_spect,1)
    
    wave_ref = np.empty(ref_spect.shape)
    for j in range(nm):
        wave_ref[j,1:-1] = np.interp(np.arange(rebin_fact*ny)/rebin_fact,np.arange(ny),wave[j,:])
        #Fill in the end wavelengths, including +/-100 km/s from the ends.
        wave_ref[j,-2] = wave_ref[j,-3] + (wave_ref[j,-3]-wave_ref[j,-4])
        wave_ref[j,0]  = wave_ref[j,1] * (3e5-1e2)/3e5
        wave_ref[j,-1] = wave_ref[j,-2] * (3e5+1e2)/3e5


rvs = np.zeros( (nf,nm) )
rv_sigs = np.zeros( (nf,nm) )
initp = np.zeros(4)
initp[0]=0.0
spect_sdev = np.sqrt(vars)
for i in range(nf):
    for j in range(nm):
        #This is the *only* non-linear interpolation function that doesn't take forever
        spline_ref = interp.InterpolatedUnivariateSpline(wave_ref[j,:], ref_spect[j,:])
        args = (wave[j,:],fluxes[i,j,:],spect_sdev[i,j,:],spline_ref)
        resid = rv_shift_resid( initp, *args)
        the_fit = op.leastsq(rv_shift_resid,initp,args=args, diag=[1e3,1e-6,1e-3,1],Dfun=rv_shift_jac,full_output=True)
        #Remove bad points...
        resid = rv_shift_resid( the_fit[0], *args)
        wbad = np.where( np.abs(resid) > 10)[0]
        args[2][wbad] = np.inf
        the_fit = op.leastsq(rv_shift_resid,initp,args=args, diag=[1e3,1e-7,1e-3,1],Dfun=rv_shift_jac, full_output=True)
        #Some testing code
        if (False):
            pp0 = np.poly1d(the_fit[0][1:])
            norm = np.exp(pp0(np.arange(ny)-ny//2))
            cc = 2.998e8
            fitted_spect = np.interp(args[0]*(1.0 + the_fit[0][0]/cc),args[3], args[4])*norm
            import pdb; pdb.set_trace()
        rvs[i,j] = the_fit[0][0]
        try:
            rv_sigs[i,j] = np.sqrt(the_fit[1][0,0])
        except:
            rv_sigs[i,j] = np.NaN
    print("Done file {0:d}".format(i))

#Plot the Barycentric corrected RVs. Note that a median over all orders is
#only a first step - a weighted mean is needed.
plt.plot(np.median(rvs,1)+((bcors[:,0] - np.median(bcors[:,0]))*1e3))
