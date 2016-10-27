"""This is a simple simulation code for GHOST or Veloce, with a class ARM that simulates
a single arm of the instrument. The key default parameters are hardwired for each named 
configuration in the __init__ function of ARM. 

Note that in this simulation code, the 'x' and 'y' directions are the along-slit and 
dispersion directions respectively... (similar to physical axes) but by convention, 
images are returned/displayed with a vertical slit and a horizontal dispersion direction.

For a simple simulation, run:

import pymfe

blue = pymfe.ghost.Arm('blue')

blue.simulate_frame()

TODO: 
1) Add spectrograph aberrations (just focus and coma)
2) Add pupil illumination plus aberrations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import optics
import os
import pdb
from polyspect import Polyspect
try:
    import pyfits
except:
    import astropy.io.fits as pyfits

class Arm(Polyspect):
    """A class for each arm of the spectrograph. The initialisation function takes a 
    series of strings representing the configuration. For GHOST, it can be "red" or "blue" 
    for the first string, and "std" or "high" for the second string. """
    
    def __init__(self,arm, mode='high'):
        self.arm=arm
        self.d = 1000/52.67   #Distance in microns
        self.theta= 65.0      #Blaze angle
        self.assym = 1.0/0.41 #Magnification
        self.gamma = 0.56     #Echelle gamma
        self.nwave = 1e2      #Wavelengths per order for interpolation.
        self.f_col = 1750.6   #Collimator focal length.
        self.lenslet_high_size = 118.0 #Lenslet flat-to-flat in microns
        self.lenslet_std_size = 197.0 #Lenslet flat-to-flat in microns
        self.microns_pix = 2.0  #When simulating the slit image, use this many microns per pixel
        self.microns_arcsec = 400.0 #Number of microns in the slit image plane per arcsec
        self.im_slit_sz = 2048 #Size of the image slit size in pixels.
        if (arm == 'red'):
            self.extra_rot = 3.0   #Additional slit rotation across an order needed to match Zemax.
            self.szx = 6144
            self.szy = 6144
            self.f_cam = 264.0
            self.px_sz = 15e-3
            self.drot = -2.0       #Detector rotation
            self.d_x = 1000/565.   #VPH line spacing
            self.theta_i=30.0      #Prism incidence angle
            self.alpha1 = 0.0      #First prism apex angle
            self.alpha2 = 0.0      #Second prism apex angle
            self.m_min = 34
            self.m_max = 67
        elif (arm == 'blue'):
            self.extra_rot = 2.0   #Additional slit rotation accross an order needed to match Zemax.
            self.szx = 4096
            self.szy = 4112
            self.f_cam = 264.0
            self.px_sz = 15e-3
            self.d_x = 1000/1137.   #VPH line spacing
            self.theta_i=30.0      #Prism incidence angle
            self.drot = -2.0         #Detector rotation. 
            self.alpha1 = 0.0      #First prism apex angle
            self.alpha2 = 0.0      #Second prism apex angle
            self.m_min = 63
            self.m_max = 95
        else:
            print("Unknown spectrograph arm!")
            raise UserWarning
        
    def set_mode(self, new_mode):
        """Set a new mode (high or standard res)
        """
        if (mode == 'high'):
            self.mode=mode
            self.lenslet_width = self.sim.lenslet_high_size
            self.nl = 28
            ## Set default profiles - object, sky and reference
            fluxes = np.zeros( (self.nl,3) )
            fluxes[2:21,0] = 0.37
            fluxes[8:15,0] = 0.78
            fluxes[11,0] = 1.0
            #NB if on the following line, fluxes[2:,1]=1.0 is set, sky will be
            #subtracted automatically.
            fluxes[2+19:,1]=1.0
            fluxes[0,2]=1.0
        elif (mode == 'std'):
            self.mode=mode
            self.lenslet_width = self.sim.lenslet_std_size
            self.nl = 17
            ## Set default profiles - object 1, sky and object 2
            fluxes = np.zeros( (self.nl,3) )
            fluxes[0:7,0]  = 1.0
            fluxes[7:10,1] = 1.0
            fluxes[10:,2] = 1.0
        else:
            print("Unknown mode!")
            raise UserWarning
            

    def make_lenslets(self,fluxes=[],mode='',seeing=0.8, llet_offset=0):
        """Make an image of the lenslets with sub-pixel sampling.
        
        Parameters
        ----------
        fluxes: float array (optional)
            Flux in each lenslet
            
        mode: string (optional)
            'high' or 'std', i.e. the resolving power mode of the spectrograph. Either
            mode or fluxes must be set.
        
        seeing: float (optional)
            If fluxes is not given, then the flux in each lenslet is defined by the seeing.
            
        llet_offset: int
            Offset in lenslets to apply to the input spectrum"""
        print("Computing a simulated slit image...")
        szx = self.im_slit_sz
        szy = 256
        fillfact = 0.98
        s32 = np.sqrt(3)/2
        hex_scale = 1.15
        conv_fwhm = 30.0  #equivalent to a 1 degree FWHM for an f/3 input ??? !!! Double-check !!!
        if len(fluxes)==28:
            mode = 'high'
        elif len(fluxes)==17:
            mode = 'std'
        elif len(mode)==0:
            mode = self.mode
        if mode=='std':
            nl=17
            lenslet_width = self.lenslet_std_size
            yoffset = (lenslet_width/self.microns_pix/hex_scale*np.array([0,-s32,s32,0,-s32,s32,0])).astype(int)
            xoffset = (lenslet_width/self.microns_pix/hex_scale*np.array([-1,-0.5,-0.5,0,0.5,0.5,1.0])).astype(int)
        elif mode=='high':
            nl=28
            lenslet_width = self.lenslet_high_size
            yoffset = (lenslet_width/self.microns_pix/hex_scale*s32*np.array([-2,2,-2,-1,-1,0,-1,-1,0,0,0,1,1,0,1,1,2,-2,2])).astype(int)
            xoffset = (lenslet_width/self.microns_pix/hex_scale*0.5*np.array([-2,0,2,-3,3,-4,-1,1,-2,0,2,-1,1,4,-3,3,-2,0,2])).astype(int)
        else:
            print("Error: mode must be standard or high")
        
        
        #Some preliminaries...
        cutout_hw = int(lenslet_width/self.microns_pix*1.5)
        im_slit = np.zeros((szy,szx))
        x = np.arange(szx) - szx/2.0
        y = np.arange(szy) - szy/2.0
        xy = np.meshgrid(x,y)
        #r and wr enable the radius from the lenslet center to be indexed
        r = np.sqrt(xy[0]**2 + xy[1]**2)
        wr = np.where(r < 2*lenslet_width/self.microns_pix)
        #g is a Gaussian used for FRD
        g = np.exp(-r**2/2.0/(conv_fwhm/self.microns_pix/2.35)**2)
        g = np.fft.fftshift(g)
        g /= np.sum(g)
        gft = np.conj(np.fft.rfft2(g))
        pix_size_slit = self.px_sz*(self.f_col/self.assym)/self.f_cam*1000.0/self.microns_pix
        pix = np.zeros( (szy,szx) )
        pix[np.where( (np.abs(xy[0]) < pix_size_slit/2) * (np.abs(xy[1]) < pix_size_slit/2) )] = 1
        pix = np.fft.fftshift(pix)
        pix /= np.sum(pix)
        pix_ft = np.conj(np.fft.rfft2(pix))
        #Create some hexagons. We go via a "cutout" for efficiency.
        h_cutout = optics.hexagon(szy, lenslet_width/self.microns_pix*fillfact/hex_scale)
        hbig_cutout = optics.hexagon(szy, lenslet_width/self.microns_pix*fillfact)
        h = np.zeros( (szy,szx) )
        hbig = np.zeros( (szy,szx) )
        h[:,szx/2-szy/2:szx/2+szy/2] = h_cutout
        hbig[:,szx/2-szy/2:szx/2+szy/2] = hbig_cutout
        if len(fluxes)!=0:
            #If we're not simulating seeing, the image-plane is uniform, and we only use
            #the values of "fluxes" to scale the lenslet fluxes. 
            im = np.ones( (szy,szx) )
            #Set the offsets to zero because we may be simulating e.g. a single Th/Ar lenslet
            #and not starlight (from the default xoffset etc)
            xoffset = np.zeros(len(fluxes),dtype=int)
            yoffset = np.zeros(len(fluxes),dtype=int)
        else:
            #If we're simulating seeing, create a Moffat function as our input profile, 
            #but just make the lenslet fluxes uniform.
            im = np.zeros( (szy,szx) )
            im_cutout = optics.moffat2d(szy,seeing*self.microns_arcsec/self.microns_pix/2, beta=4.0)
            im[:,szx/2-szy/2:szx/2+szy/2] = im_cutout
            fluxes = np.ones(len(xoffset))
            
        #Go through the flux vector and fill in each lenslet.
        for i in range(len(fluxes)):
            im_one = np.zeros((szy,szx))
            im_cutout = np.roll(np.roll(im,yoffset[i],axis=0),xoffset[i],axis=1)*h
            im_cutout = im_cutout[szy/2-cutout_hw:szy/2+cutout_hw,szx/2-cutout_hw:szx/2+cutout_hw]
            prof = optics.azimuthalAverage(im_cutout, returnradii=True, binsize=1)
            prof = (prof[0],prof[1]*fluxes[i])
            xprof = np.append(np.append(0,prof[0]),np.max(prof[0])*2)
            yprof = np.append(np.append(prof[1][0],prof[1]),0)
            im_one[wr] = np.interp(r[wr], xprof, yprof)
            im_one = np.fft.irfft2(np.fft.rfft2(im_one)*gft)*hbig
            im_one = np.fft.irfft2(np.fft.rfft2(im_one)*pix_ft)
            #!!! The line below could add tilt offsets... important for PRV simulation !!!
            #im_one = np.roll(np.roll(im_one, tilt_offsets[0,i], axis=1),tilt_offsets[1,i], axis=0)*hbig
            the_shift = int( (llet_offset + i - nl/2.0)*lenslet_width/self.microns_pix )
            im_slit += np.roll(im_one,the_shift,axis=1)
        return im_slit
                
    def simulate_image(self,x,w,b,matrices,im_slit,spectrum=[],nx=0, xshift=0.0, yshift=0.0, rv=0.0):
            """Simulate a spectrum on the CCD.
            
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
        """Simulate a single frame. 
        
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
            print "ERROR: unknown mode."
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
