"""This is a simple simulation and extraction definition code for RHEA. The key is to
use Tobias's spectral fitting parameters and define the same kinds of extraction arrays
as pyghost.
"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import optics
import os
import scipy.optimize as op
import pdb
from polyspect import Polyspect
try:
    import pyfits
except:
    import astropy.io.fits as pyfits

class Format(Polyspect):
    """A class for each possible configuration of the spectrograph. The initialisation 
    function takes a single string representing the configuration.
    
    sim must include (!!! i.e. it should inherit and over-ride default generic modules):
    
    spectral_format_with_matrix()
    make_lenslets()
    
    fluxes (nlenslets x nobj) array
    nl (nlenslets)
    szx (size in x [non-dispersion] direction)
    mode (string,for error messages)
    lenslet_width, im_slit_sz, microns_pix (together define the make_lenslets output)
    
    """
    
    def __init__(self,spect='rhea2',mode="single",m_min=None,m_max=None):
        self.spect=spect
        if (spect == 'rhea2'):
            self.szy   = 2200
            self.szx   = 1375         ## NB, this fixes a 2x1 binning.
            self.xbin  = 2
            self.m_min = 96
            self.m_max = 129
            self.m_ref = 112        ## The reference order.
            self.extra_rot = 1.0    ## Extra rotation in degrees
            self.transpose = False  ## Do we transpose raw fits files
        elif (spect == 'subaru'):
            self.szy = 2750
            self.szx = 2200
            self.xbin = 1
            self.m_min = 68
            self.m_max = 96
            if m_min:
                self.m_min = m_min
            if m_max:
                self.m_max = m_max
            self.m_ref = 82
            self.extra_rot = 1.0
            self.transpose = True
        else:
            print("Unknown spectrograph arm!")
            raise UserWarning
        self.rnoise = 20.0
        self.gain=0.3 #Gain in electrons per DN. 
        ## Some slit parameters...
        self.mode       = mode
        if (mode=="single"):
            self.nl         = 1
            self.im_slit_sz = 64            #Number of pixels for a simulated slit image
        elif (mode=="slit"):
            self.nl=10
            self.im_slit_sz = 600 #Need 250/10*11*2 on each side! Crazy.
        self.microns_pix = 10.0      #Microns per pixel in the simulated image
        self.lenslet_width=250.0        #Width of a single lenslet in microns
        self.fib_image_width_in_pix = 3.5
        self.fluxes = np.eye( self.nl )  
        ## And, critically, the slit-to-pixel scaling for the first and last order.
        self.slit_microns_per_det_pix_first = 71.0 # WAS 62.5, based on optical diagram.
        self.slit_microns_per_det_pix_last = 65.0 # WAS 62.5, based on optical diagram.
               
        
    def make_lenslets(self,fluxes=[]):
        """Make an image of the lenslets with sub-pixel sampling.
        
        Parameters
        ----------
        fluxes: float array (optional)
            Flux in each lenslet
            
        mode: string (optional)
            'subaru' or 'mso', i.e. the input type of the spectrograph. Either
            mode or fluxes must be set.
        
        llet_offset: int
            Offset in lenslets to apply to the input spectrum"""
        ## In this case, we obviously ignore the fluxes!
        x = (np.arange(self.im_slit_sz) - self.im_slit_sz/2)*self.microns_pix
        xy = np.meshgrid(x,x)
        ## A simple Gaussian approximation to the fiber far field (or near-field in the
        ## case of the original RHEA2. 2.35482 scales FWHM to stdev
        slit_microns_per_det_pix = 0.5*(self.slit_microns_per_det_pix_first + self.slit_microns_per_det_pix_last)
        gsig = self.fib_image_width_in_pix*slit_microns_per_det_pix/2.35482
        sim_im0 = np.exp( - xy[0]**2/2.0/(gsig/self.xbin)**2 - xy[1]**2/2.0/gsig**2  )
        sim_im = sim_im0.copy()
        if (self.mode == "slit"):
            #If fluxes isn't given, use the default
            if len(fluxes) != 0:
                sim_im *= fluxes[0]
                for i in range(1,len(fluxes)):
                    sim_im += fluxes[i]*np.roll(sim_im0,int((i+1)*self.lenslet_width/self.microns_pix),axis=1)
        elif (self.mode != "single"):
            print("Error: invalid mode " + self.mode)
            raise UserWarning
        return sim_im
                
    def simulate_image(self,x,w,b,matrices,im_slit,spectrum=[],nx=0, xshift=0.0, yshift=0.0, rv=0.0):
            """Simulate a spectrum on the CCD, for a single source. 
            WARNING: from pyghost - not implemented yet!
            
            Parameters
            ----------
            x,w,b,matrices: float arrays
                See the output of spectral_format_with_matrix
            im_slit: float array
                See the output of make_lenslets
            spectrum: (2,nwave) array (optional)
                An input spectrum, arbitrarily gridded (but at a finer resolution than the 
                spectrograph resolving power. If not given, a solar spectrum is used.
            nx: float
                Number of x (along-slit) direction pixels in the image. If not given or
                zero, a square CCD is assumed.
            xshift: float
                Bulk shift to put in to the spectrum along the slit.
            yshift: float
                NOT IMPLEMENTED
            rv: float
                Radial velocity in m/s.
            """
            #If no input spectrum, use the sun.
            if len(spectrum)==0:
                d =pyfits.getdata(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../data/ardata.fits.gz'))
                spectrum=np.array([np.append(0.35,d['WAVELENGTH'])/1e4,np.append(0.1,d['SOLARFLUX'])])
            nm = x.shape[0]
            ny = x.shape[1]
            if nx==0:
                nx = ny
            image = np.zeros( (ny,nx) )
            #Simulate the slit image within a small cutout region.
            cutout_xy = np.meshgrid( np.arange(81)-40, np.arange(7)-3 )
            #Loop over orders
            for i in range(nm):
                for j in range(ny):
                    if x[i,j] != x[i,j]:
                        continue
                    #We are looping through y pixel and order. The x-pixel is therefore non-integer.
                    #Allow an arbitrary shift of this image.
                    the_x = x[i,j] + xshift
                    #Create an (x,y) index of the actual pixels we want to index.
                    cutout_shifted = (cutout_xy[0].copy() + int(the_x) + nx/2, \
                                      cutout_xy[1].copy() + j)
                    ww = np.where( (cutout_shifted[0]>=0) * (cutout_shifted[1]>=0) *  \
                                   (cutout_shifted[0]<nx) * (cutout_shifted[1]<ny) )
                    cutout_shifted = (cutout_shifted[0][ww], cutout_shifted[1][ww])
                    flux = np.interp(w[i,j]*(1 + rv/299792458.0),spectrum[0], spectrum[1],left=0,right=0)
                    #Rounded to the nearest microns_pix, find the co-ordinate in the simulated slit image corresponding to 
                    #each pixel. The co-ordinate order in the matrix is (x,y).
                    xy_scaled = np.dot( matrices[i,j], np.array([cutout_xy[0][ww]+int(the_x)-the_x,cutout_xy[1][ww]])/self.microns_pix ).astype(int)
                    image[cutout_shifted[1],cutout_shifted[0]] += b[i,j]*flux*im_slit[xy_scaled[1] + im_slit.shape[0]/2,xy_scaled[0] + im_slit.shape[1]/2]
                print('Done order: {0}'.format(i + self.m_min))
            return image
        
    def simulate_frame(self, output_prefix='test_', xshift=0.0, yshift=0.0, rv=0.0, 
        rv_thar=0.0, flux=1e2, rnoise=3.0, gain=1.0, use_thar=True, mode='high', return_image=False, thar_flatlamp=False):
        """Simulate a single frame, including Thorium/Argon or Xenon reference.
        WARNING: from pyghost - not implemented yet!
        
        TODO (these can be implemented manually using the other functions): 
        1) Variable seeing (the slit profile is currently fixed)
        2) Standard resolution mode.
        3) Sky
        4) Arbitrary input spectra 
        
        Parameters
        ----------
        output_prefix: string (optional)
            Prefix for the output filename.
        
        xshift: float (optional)
            x-direction (along-slit) shift. 
        
        yshift: float (optional)
            y-direction (spectral direction) shift.
        
        rv: float (optional)
            Radial velocity in m/s for the target star with respect to the observer.
            
        rv_thar: float (optional)
            Radial velocity in m/s applied to the Thorium/Argon source.  It is unlikely 
            that this is useful (use yshift instead for common shifts in the dispersion
            direction).
            
        flux: float (optional)
            Flux multiplier for the reference spectrum to give photons/pix.
            
        rnoise: float (optional)
            Readout noise in electrons/pix
        
        gain: float (optional)
            Gain in electrons per ADU.
        
        use_thar: bool (optional)
            Is the Thorium/Argon lamp in use?
            
        mode: string (optional)
            Can be 'high' or 'std' for the resolution mode.
            
        return_image: bool (optional)
            Do we return an image as an array? The fits file is always written.
        """
        x,w,b,matrices = self.spectral_format_with_matrix()

        if (mode == 'high'):
            slit_fluxes = np.ones(19)*0.37
            slit_fluxes[6:13] = 0.78
            slit_fluxes[9] = 1.0
            slit_fluxes /= np.mean(slit_fluxes)
            im_slit = self.make_lenslets(fluxes=slit_fluxes, mode='high', llet_offset=2)
            image = self.simulate_image(x,w,b,matrices,im_slit, xshift=xshift,rv=rv)
            
            if (use_thar):
                #Create an appropriately convolved Thorium-Argon spectrum after appropriately
                #convolving.
                thar = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../data/mnras0378-0221-SD1.txt'),usecols=[0,1,2])
                thar_wave = 3600 * np.exp(np.arange(5e5)/5e5)
                thar_flux = np.zeros(5e5)
                ix = (np.log(thar[:,1]/3600)*5e5).astype(int)
                ix = np.minimum(np.maximum(ix,0),5e5-1).astype(int)
                thar_flux[ ix ] = 10**(np.minimum(thar[:,2],4))
                thar_flux = np.convolve(thar_flux,[0.2,0.5,0.9,1,0.9,0.5,0.2],mode='same')
                #Make the peak flux equal to 10
                thar_flux /= 0.1*np.max(thar_flux)
                #Include an option to assume the Th/Ar fiber is connected to a flat lamp
                if thar_flatlamp:
                    thar_flux[:]=10
                thar_spect = np.array([thar_wave/1e4,thar_flux])
                #Now that we have our spectrum, create the Th/Ar image.
                slit_fluxes = np.ones(1)
                im_slit2 = self.make_lenslets(fluxes=slit_fluxes, mode='high', llet_offset=0)
                image += self.simulate_image(x,w,b,matrices,im_slit2, spectrum=thar_spect, xshift=xshift,rv=rv_thar)
        else:
            print("ERROR: unknown mode.")
            raise UserWarning

        #Prevent any interpolation errors (negative flux) prior to adding noise.
        image = np.maximum(image,0)
        image = np.random.poisson(flux*image) + rnoise*np.random.normal(size=image.shape)
        #For conventional axes, transpose the image, and divide by the gain in e/ADU
        image = image.T/gain
        #Now create our fits image!
        hdu = pyfits.PrimaryHDU(image)
        hdu.writeto(output_prefix + self.arm + '.fits', clobber=True)
        
        if (return_image):
            return image
