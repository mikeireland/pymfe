"""This module/class contains functionality for computing (and plotting) radial 
velocities and creating reference spectra for extracted fluxes. This should 
ideally remain independent of the extraction method, such that it does not 
matter which spectrograph took the data, nor what "Spectrograph" object was
used for extraction.

Most of the code below has been moved from the script "test_rhea2_extract.py".
Work still needs to be done post-refactor to ensure function input and outputs
are sensible, their docstrings are informative and they follow the principles of
Object Oriented Programming - such as the Single Responsibility Principle (Along
with a general clean up of the code and comments, such as having the code meet 
the python line length guidelines --> the main benefit of which is having 
multiple editors open side by side on smaller screens)

TODO
----
1) Move extract method to either extract module or rhea
2) Try to separate calculation/processing of data from saving/loading/displaying
3) Tidy up inputs to functions (e.g. cull unnecessary input parameters)
4) Make create_ref_spect() output variances (Median Absolute Deviations)
5) Possibly have dark calibration (for both flats and science frames) in its own
   method. This would clean up the existing extract method, removing the need
   to check whether darks and flats had been passed in (or varying permutations
   of each - e.g. in the case where some of the data has already been dark 
   corrected, such as the solar data)
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.interpolate as interp
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import constants as const
import PyAstronomy.pyasl as pyasl
import opticstools as ot
try:
    import pyfits
except:
    import astropy.io.fits as pyfits

class RadialVelocity():
    """A RadialVelocity object for calculating and plotting RVS and generating
    reference spectra.
    
    Unclear if the object needs to be initialised with any parameters at this 
    stage. Perhaps a file path?
    """
    
    def __init__(self):
        """(Presently empty) constructor. 
        """
        pass
        
    def rv_shift_resid(self, params, wave, spect, spect_sdev, spline_ref, 
                       return_spect=False):
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
        # Lets get this sign correct. A redshift (positive velocity) means that
        # a given wavelength for the reference corresponds to a longer  
        # wavelength for the target, which in turn means that the target 
        # wavelength has to be interpolated onto shorter wavelengths for the 
        # reference.
        fitted_spect = spline_ref(wave*(1.0 - params[0]/const.c.si.value))*norm
        
        if return_spect:
            return fitted_spect
        else:
            return (fitted_spect - spect)/spect_sdev

    def rv_shift_jac(self, params, wave, spect, spect_sdev, spline_ref):
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
            The Jacobian.
        """
        ny = len(spect)
        xx = np.arange(ny)-ny//2
        norm = np.exp(params[1]*xx**2 + params[2]*xx + params[3])
        fitted_spect = spline_ref(wave*(1.0 - params[0]/const.c.si.value))*norm
        jac = np.empty( (ny,4) )
        jac[:,3] = fitted_spect/spect_sdev
        jac[:,2] = fitted_spect*xx/spect_sdev
        jac[:,1] = fitted_spect*xx**2/spect_sdev
        jac[:,0] = (spline_ref(wave*(1.0 - (params[0] + 1.0)/const.c.si.value))*
                    norm - fitted_spect)/spect_sdev
        return jac

    def create_ref_spect(self, wave, fluxes, vars, bcors, rebin_fact=2, 
                         gauss_sdev=1.0, med_cut=0.6,gauss_hw=7):
        """Create a reference spectrum from a series of target spectra, after 
        correcting the spectra barycentrically. 
        
        Parameters
        ----------
        wave: 2D np.array(float)
            Wavelength coordinate map of form (Order, Wavelength/pixel)
        fluxes: 3D np.array(float)
            Fluxes of form (Observation, Order, Flux/pixel)
        vars: 3D np.array(float)
            Variance of form (Observation, Order, Variance/pixel)
        bcors: 1D np.array(float)
            Barycentric correction for each observation.
        rebin_fact: int
            Factor by which to rebin.
        gauss_sdev:
            ...
        med_cut:
            ...
        gauss_hw:
            ...
        
        Returns
        -------
        wave_ref: 2D np.array(float)
            Wavelength coordinate map of form (Order, Wavelength/pixel*2+2), 
            where the wavelength scale has been interpolated.
        ref_spect: 2D np.array(float)
            Reference spectrum of form (Order, Flux/pixel*2+2), 
            where the flux scale has been interpolated.
        """
        nm = fluxes.shape[1]
        ny = fluxes.shape[2]
        nf = fluxes.shape[0]
        
        C = const.c.si.value

        #Create arrays for our outputs.
        wave_ref = np.empty( (nm,rebin_fact*ny + 2) )
        ref_spect = np.empty( (nm,rebin_fact*ny + 2) )

        #First, rebin everything.
        new_shape = (fluxes.shape[1],rebin_fact*fluxes.shape[2])
        fluxes_rebin = np.empty( (fluxes.shape[0],fluxes.shape[1],
                                  rebin_fact*fluxes.shape[2]) )
        for i in range(nf):
            fluxes_rebin[i] = ot.utils.regrid_fft(fluxes[i],new_shape)

        #Create the final wavelength grid.    
        for j in range(nm):
            wave_ref[j,1:-1] = np.interp(np.arange(rebin_fact*ny)/rebin_fact, 
                                         np.arange(ny),wave[j,:])
            #Fill in the end wavelengths, including +/-100 km/s from the ends.
            wave_ref[j,-2] = wave_ref[j,-3] + (wave_ref[j,-3]-wave_ref[j,-4])
            wave_ref[j,0]  = wave_ref[j,1] * (C + 1e5)/C
            wave_ref[j,-1] = wave_ref[j,-2] * (C - 1e5)/C

        #Barycentric correct
        for i in range(nf):
            for j in range(nm):
                # Awkwardly, we've extended the wavelength scale by 2 elements,  
                # but haven't yet extended the fluxes...
                ww = wave_ref[j,1:-1]
                fluxes_rebin[i,j] = np.interp(ww*(1-bcors[i]/C), ww[::-1],
                                              fluxes_rebin[i,j,::-1])
                
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
            
        # Create a Gaussian smoothing function for the reference spectrum. This 
        # is needed to prevent a bias to zero radial velocity, especially in the 
        # case of few data points.
        gg = np.exp(-(np.arange(2*gauss_hw+1)-gauss_hw)**2/2.0/gauss_sdev**2)
        gg /= np.sum(gg)
        one_order = np.empty(flux_ref.shape[1] + 2*gauss_hw)
        for j in range(nm):
            one_order[gauss_hw:-gauss_hw] = flux_ref[j,:]
            one_order[:gauss_hw] = one_order[gauss_hw]
            one_order[-gauss_hw:] = one_order[-gauss_hw-1]
            ref_spect[j,:] = np.convolve(one_order, gg, 
                                         mode='same')[gauss_hw-1:1-gauss_hw]
        
        return wave_ref, ref_spect

    def extract_spectra(self, files, extractor, star_dark=None, flat_files=None,   
                        flat_dark=None, location=('151.2094','-33.865',100.0),  
                        coord=None, do_bcor=True):
        """Extract the spectrum from a file, given a dark file, a flat file and
        a dark for the flat. 
        
        TODO:
        Not the neatest implementation, but should account for the fact that
        there are no flats or darks for the ThAr frames. Might be worth tidying
        up and making the implementation a little more elegant.
        
        Parameters
        ----------
        files: list of strings
            One string for each file. CAn be on separate nights - a full 
            pathname should be given.
        star_dark:
            
        flat_files: list of strings.
            One string for each star file. CAn be on separate nights - a full 
            pathname should be given.
        flat_dark:
            
        location: (lattitude:string, longitude:string, elevation:string)
            The location on Earth where the data were taken.
        coord: astropy.coordinates.sky_coordinate.SkyCoord
            The coordinates of the observation site
        do_bcor: boolean
            Flag for whether to do barycentric correction
        
        Returns
        -------
        fluxes: 3D np.array(float)
            Fluxes of form (Observation, Order, Flux/pixel)
        vars: 3D np.array(float)
            Variance of form (Observation, Order, Variance/pixel)
        bcors: 1D np.array(float)
            Barycentric correction for each observation.
        wave: 2D np.array(float)
            Wavelength coordinate map of form (Order, Wavelength/pixel)
        mjds: 1D np.array(float)
            Modified Julian Date (MJD) of each observation.
        """
        # Initialise list of return values 
        # Each index represents a single observation
        fluxes = []
        vars = []
        dates = []
        bcors = []
        
        #!!! This is dodgy, as files and flat_files should go together in a dict
        for ix,file in enumerate(files):
            # Dark correct the science and flat frames
            # Only if flat/darks have been supplied --> ThAr might not have them
            # If not supplied, just use science/reference data
            try:
                if star_dark and flat_files and flat_dark:
                    data = pyfits.getdata(file) - star_dark
                    flat = pyfits.getdata(flat_files[ix]) - flat_dark
                else:
                    data = pyfits.getdata(file)
            except: 
                print('Unable to calibrate file ' + file + 
                      '. Check that format of data arrays are consistent.')
                continue            
            header = pyfits.getheader(file)
            
            date = Time(header['DATE-OBS'], location=location)
            dates.append(date)
            
            # Determine the barycentric correction
            if do_bcor:
                if not coord:
                    coord=SkyCoord(ra=float(header['RA']), 
                                   dec=float(header['DEC']) , unit='deg')
                if not location:
                    location=(float(header['LONG']), float(header['LAT']), 
                              float(header['HEIGHT']))
                #(obs_long, obs_lat, obs_alt, ra2000, dec2000, jd, debug=False)
                bcors.append(1e3*pyasl.helcorr(float(location[0]), 
                             float(location[1]),location[2],coord.ra.deg, 
                             coord.dec.deg,date.jd)[0] )
            else:
                bcors.append(0.0)
            
            # Extract the fluxes and variance for the science and flat frames
            flux, var = extractor.one_d_extract(data=data, rnoise=20.0)
            
            # Continue only when flats and darks have been supplied
            # Perform flat field correction and adjust variances
            if star_dark and flat_files and flat_dark:
                flat_flux, fvar = extractor.one_d_extract(data=flat, 
                                                          rnoise=20.0)
            
                for j in range(flat_flux.shape[0]):
                    medf = np.median(flat_flux[j])
                    flat_flux[j] /= medf
                    fvar[j] /= medf**2
                
                #Calculate the variance after dividing by the flat
                var = var/flat_flux**2 + fvar * flux**2/flat_flux**4
                
                #Now normalise the flux.
                flux /= flat_flux

            # Regardless of whether the data has been flat field corrected, 
            # append to the arrays and continue
            fluxes.append(flux[:,:,0])
            vars.append(var[:,:,0])

        fluxes = np.array(fluxes)
        vars = np.array(vars)
        bcors = np.array(bcors)
        mjds = np.array([d.mjd for d in dates])

        return fluxes, vars, bcors, mjds

    def calculate_rv_shift(self, wave_ref, ref_spect, fluxes, vars, bcors, 
                           wave):
        """Calculates the Radial Velocity shift. 
        
        Parameters
        ----------
        wave_ref: 2D np.array(float)
            Wavelength coordinate map of form (Order, Wavelength/pixel*2+2), 
            where the wavelength scale has been interpolated.
        ref_spect: 2D np.array(float)
            Reference spectrum of form (Order, Flux/pixel*2+2), 
            where the flux scale has been interpolated.
        fluxes: 3D np.array(float)
            Fluxes of form (Observation, Order, Flux/pixel)
        vars: 3D np.array(float)
            Variance of form (Observation, Order, Variance/pixel)    
        bcors: 1D np.array(float)
            Barycentric correction for each observation.
        wave: 2D np.array(float)
            Wavelength coordinate map of form (Order, Wavelength/pixel)

        Returns
        -------
        rvs: 2D np.array(float)
            Radial velocities of format (Observation, Order)
        rv_sigs: 2D np.array(float)
            Radial velocity sigmas of format (Observation, Order)
        """
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
            # Start with initial guess of no intrinsic RV for the target.
            initp[0] = -bcors[i] 
            
            for j in range(nm):
                # This is the *only* non-linear interpolation function that 
                # doesn't take forever
                spl_ref = interp.InterpolatedUnivariateSpline(wave_ref[j,::-1], 
                                                              ref_spect[j,::-1])
                args = (wave[j,:], fluxes[i,j,:], spect_sdev[i,j,:], spl_ref)
                
                # Remove edge effects in a slightly dodgy way. 
                # 20 pixels is about 30km/s. 
                args[2][:20] = np.inf
                args[2][-20:] = np.inf
                the_fit = op.leastsq(self.rv_shift_resid, initp, args=args, 
                                     diag=[1e3,1e-6,1e-3,1], 
                                     Dfun=self.rv_shift_jac, full_output=True)
                
                #Remove bad points...
                resid = self.rv_shift_resid( the_fit[0], *args)
                wbad = np.where( np.abs(resid) > 7)[0]
                args[2][wbad] = np.inf
                the_fit = op.leastsq(self.rv_shift_resid, initp,args=args, 
                                     diag=[1e3,1e-7,1e-3,1], 
                                     Dfun=self.rv_shift_jac, full_output=True)
                
                #Some outputs for testing
                fitted_spects[i,j] = self.rv_shift_resid(the_fit[0], *args, 
                                                         return_spect=True)
                #Save the fit and the uncertainty.
                rvs[i,j] = the_fit[0][0]
                try:
                    rv_sigs[i,j] = np.sqrt(the_fit[1][0,0])
                except:
                    rv_sigs[i,j] = np.NaN
            print("Done file {0:d}".format(i))
            
        return rvs, rv_sigs
        
    def save_fluxes(self, files, fluxes, vars, bcors, wave, mjds, out_path):
        """Method to save the extracted spectra.
        
        TODO:
        Might want to remove the dependence on files (to get the headers) as it
        will prevent (or complicate) the saving of the reference spectrum.
        
        Parameters
        ----------
        fluxes: 3D np.array(float)
            Fluxes of form (Observation, Order, Flux/pixel)
        vars: 3D np.array(float)
            Variance of form (Observation, Order, Variance/pixel)
        bcors: 1D np.array(float)
            Barycentric correction for each observation.
        wave: 2D np.array(float)
            Wavelength coordinate map of form (Order, Wavelength/pixel)
        mjds: 1D np.array(float)
            Modified Julian Date (MJD) of each observation.
        out_path: String
            The directory to save the extracted fluxes.
        """
        # Loop through each extracted spectrum
        for i, file in enumerate(files):
            # Extract the header information from the file
            header = pyfits.getheader(file)
            
            file_name = file.split("\\")[-1].split(".")[0] + "_extracted.fits"
            
            full_path = out_path + file_name
            
            # Save to fits
            hl = pyfits.HDUList()
            hl.append(pyfits.ImageHDU(fluxes[i], header))
            hl.append(pyfits.ImageHDU(vars[i]))
            hl.append(pyfits.ImageHDU(wave))
            col1 = pyfits.Column(name='bcor', format='D', 
                                 array=np.array([bcors[i]]))
            col2 = pyfits.Column(name='mjd', format='D', 
                                 array=np.array([mjds[i]]))
            cols = pyfits.ColDefs([col1, col2])
            hl.append(pyfits.new_table(cols))
            hl.writeto(full_path, clobber=True)
            #hl.flush()
    
    def save_ref_spect(self, files, ref_spect, vars_ref, wave_ref, bcors, mjds, 
                       out_path):
        """Method to save an extracted reference spectrum
        
        Parameters
        ----------
        ref_spect: 3D np.array(float)
            Fluxes of form (Observation, Order, Flux/pixel)
        vars_ref: 3D np.array(float)
            Variance of form (Observation, Order, Variance/pixel)
        wave_ref: 2D np.array(float)
            Wavelength coordinate map of form (Order, Wavelength/pixel)
        bcors: 1D np.array(float)
            Barycentric correction for each observation used to create ref_spect
        mjds: 1D np.array(float)
            Modified Julian Date (MJD) of each observation used to create 
            ref_spect
        out_path: String
            The directory to save the reference spectrum
        """
        header = pyfits.header.Header()
        full_path = out_path + "reference_spectrum_" + str(len(files)) + ".fits"
        
        # Record which spectra were used to create the reference
        for i, file in enumerate(files):
            # Extract the file name of each file and store in the header
            file_name = file.split("\\")[-1].split(".")[0] + "_extracted.fits"
            header_name = "COMB" + str(i)
            comment = "Combined spectrum #" + str(i)
            header[header_name] = (file_name, comment)
            
        # Save to fits
        hl = pyfits.HDUList()
        hl.append(pyfits.ImageHDU(ref_spect, header))
        hl.append(pyfits.ImageHDU(vars_ref[0]))
        hl.append(pyfits.ImageHDU(wave_ref))
        col1 = pyfits.Column(name='bcor', format='D', array=np.array([bcors[0]]))
        col2 = pyfits.Column(name='mjd', format='D', 
                             array=np.array([mjds[0]]))
        cols = pyfits.ColDefs([col1, col2])
        hl.append(pyfits.new_table(cols))
        hl.writeto(full_path, clobber=True)
    
    def load_ref_spect(self, path):
        """Method to load a previously saved reference spectrum
        
        Parameters
        ----------
        path: string
            The file path to the saved reference spectrum.
            
        Returns
        -------
        ref_spect: 3D np.array(float)
            Fluxes of form (Observation, Order, Flux/pixel)
        vars_ref: 3D np.array(float)
            Variance of form (Observation, Order, Variance/pixel)
        wave_ref: 2D np.array(float)
            Wavelength coordinate map of form (Order, Wavelength/pixel)
        bcors_ref: 1D np.array(float)
            Barycentric correction for each observation used to create ref_spect
        mjds_ref: 1D np.array(float)
            Modified Julian Date (MJD) of each observation used to create 
            ref_spect
        """
        hl = pyfits.open(path)
        ref_spect = hl[0].data
        vars_ref = hl[1].data
        wave_ref = hl[2].data
        bcors_ref = hl[3].data['bcor'][0]
        mjds_ref = hl[3].data['mjd'][0]
        hl.close()
        
        return ref_spect, vars_ref, wave_ref, bcors_ref, mjds_ref
    
    def load_fluxes(self, files):
        """Loads previously saved fluxes.
        
        Parameters
        ----------
        files: [string]
            String list of filepaths of the saved fluxes
            
        Returns
        -------
        fluxes: 3D np.array(float)
            Fluxes of form (Observation, Order, Flux/pixel)
        vars: 3D np.array(float)
            Variance of form (Observation, Order, Variance/pixel)
        bcors: 1D np.array(float)
            Barycentric correction for each observation.
        wave: 2D np.array(float)
            Wavelength coordinate map of form (Order, Wavelength/pixel)
        mjds: 1D np.array(float)
            Modified Julian Date (MJD) of each observation.
        """
        fluxes = []
        vars = []
        wave = []
        bcors = []
        mjds = []
        
        for f in files:
            hl = pyfits.open(f)
            fluxes.append(hl[0].data)
            vars.append(hl[1].data)
            wave = hl[2].data # Only need one (assumption of same instrument)
            bcors.append(hl[3].data['bcor'][0])
            mjds.append(hl[3].data['mjd'][0])
            hl.close()
            
        fluxes = np.array(fluxes)
        vars = np.array(vars)
        #wave = np.array(hl[2].data) 
        bcors = np.array(bcors)
        mjds = np.array(mjds)    
        
        return fluxes, vars, wave, bcors, mjds
    
    def plot_rvs(self, rvs, rv_sigs, mjds, dates, bcors, plot_title):
        """Plots the barycentrically corrected Radial Velocities.       
        
        Note:
        Not complete.
        
        Parameters
        ----------
        rvs: 2D np.array(float)
            Radial velocities of format (Observation, Order)
        rv_sigs: 2D np.array(float)
            Radial velocity sigmas of format (Observation, Order)
        mjds: 1D np.array(float)
            Modified Julian Date (MJD) of each observation.
        bcors: 1D np.array(float)
            Barycentric correction for each observation.
        plot_title: String
            Name of the plot
        """
        # Dimensions (Number of observations and orders respectively)
        nf = rvs.shape[0]
        nm = rvs.shape[1]
        
        # Plot the Barycentric corrected RVs. Note that a median over all orders 
        # is only a first step - a weighted mean is needed.
        plt.clf()

        rvs += bcors.repeat(nm).reshape( (nf,nm) )

        rv_mn, wt_sum = np.average(rvs,axis=1, weights=1.0/rv_sigs**2, 
                                   returned=True)
        rv_mn_sig = 1.0/np.sqrt(wt_sum)
        rv_med1 = np.median(rvs,1)
        rv_med2 = np.median(rvs[:,3:20],1)

        #plt.plot_date([dates[i].plot_date for i in range(len(dates))], rv_mn)
        #plt.errorbar(mjds, rv_mn, yerr=rv_mn_sig,fmt='o')
        
        plt.errorbar(mjds, rv_med2, yerr=rv_mn_sig,fmt='o')
        plt.xlabel('Date (MJD)')
        plt.ylabel('Barycentric RV (m/s)')
        plt.title(plot_title)
        
        plt.plot_date([dates[i].plot_date for i in range(len(dates))], rv_mn)
        plt.show()
        
    def save_rvs(self, rvs, rv_sigs, mjds, base_save_path):
        """Method for saving calculated radial velocities and their errors to
        csv files.
        
        Parameters
        ----------
        wave_ref: 2D np.array(float)
            Wavelength coordinate map of form (Order, Wavelength/pixel*2+2), 
            where the wavelength scale has been interpolated.
        ref_spect: 2D np.array(float)
            Reference spectrum of form (Order, Flux/pixel*2+2), 
            where the flux scale has been interpolated.
        mjds: 1D np.array(float)
            Modified Julian Date (MJD) of each observation.
        base_save_path: string
            The base of each of the csv file paths.
        """
        # Dimensions (Number of observations and orders respectively)
        nf = rvs.shape[0]
        nm = rvs.shape[1]
        
        # Setup save paths
        rv_file = base_save_path + "_rvs.csv"
        rv_sig_file = base_save_path + "_rv_sig.csv"
        
        # Headers for each csv
        rv_h = "RV in m/s for each order, for each MJD epoch"
        rv_sig_h = "RV uncertainties in m/s for each order, for each MJD epoch"
        
        # Save rvs and errors
        np.savetxt(rv_file, np.append(mjds.reshape(nf,1), rvs,axis=1), 
                   fmt="%10.4f" + nm*", %6.1f", header=rv_h)
        np.savetxt(rv_sig_file, np.append(mjds.reshape(nf,1),rv_sigs,axis=1), 
                   fmt="%10.4f" + nm*", %6.1f", header=rv_sig_h)