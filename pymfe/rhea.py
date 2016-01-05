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
try:
    import pyfits
except:
    import astropy.io.fits as pyfits

class Format():
    """A class for each possible configuration of the spectrograph. The initialisation 
    function takes a single string representing the configuration.
    
    sim must include:
    
    spectral_format_with_matrix()
    make_lenslets()
    
    fluxes (nlenslets x nobj) array
    nl (nlenslets)
    szx (size in x [non-dispersion] direction)
    mode (string,for error messages)
    lenslet_width, im_slit_sz, microns_pix (together define the make_lenslets output)
    
    """
    
    def __init__(self,spect='rhea2',mode="single"):
        self.spect=spect
        if (spect == 'rhea2'):
            self.szy   = 2200
            self.szx   = 1375         ## NB, this fixes a 2x1 binning.
            self.xbin  = 2
            self.m_min = 96
            self.m_max = 129
            self.m_ref = 112        ## The reference order.
            self.extra_rot = 1.0    ## Extra rotation in degrees
        else:
            print("Unknown spectrograph arm!")
            raise UserWarning
        ## Some slit parameters...
        self.mode       = mode
        self.nl         = 1
        self.im_slit_sz = 64            #Number of pixels for a simulated slit image
        self.microns_pix = 10.0      #Microns per pixel in the simulated image
        self.lenslet_width=250.0        #Width of a single lenslet in microns
        self.fib_image_width_in_pix = 4.0
        self.fluxes = np.ones( (1,1) )  #Default fluxes for simulated data
        ## And, critically, the slit-to-pixel scaling
        self.slit_microns_per_det_pix = 62.5 # WAS 62.5
        
    def wave_fit_resid(self, params, ms, waves, ys, ydeg=3, xdeg=3):
        """A fit function for read_lines_and_fit (see that function for details), to be 
        used in scipy.optimize.leastsq. The same function is used in fit_to_x, but 
        in that case "waves" is replaced by "xs".
        """
        if np.prod(params.shape) != (xdeg+1)*(ydeg+1):
            print("Parameters are flattened - xdeg and ydeg must be correct!")
            raise UserWarning
        params = params.reshape( (ydeg+1,xdeg+1) )
        if (len(ms) != len(ys)):
            print("ms and ys must all be the same length!")
            raise UserWarning
        mp = self.m_ref/ms - 1
        ps = np.empty( (len(ms), ydeg+1) )
        #Find the polynomial coefficients for each order.
        for i in range(ydeg+1):
            polyq = np.poly1d(params[i,:])
            ps[:,i] = polyq(mp)
        wave_mod = np.empty( len(ms) ) 
#        pdb.set_trace()
        for i in range(len(ms)):
            polyp = np.poly1d(ps[i,:])
            wave_mod[i] = polyp(ys[i]-self.szy/2)
        return wave_mod - waves
        
    def read_lines_and_fit(self, init_mod_file='', pixdir='',outdir='./', ydeg=3, xdeg=3):
        """Read in a series of text files that have a (Wavelength, pixel) format and file names
        like order99.txt and order100.txt. Fit an nth order polynomial to the wavelength
        as a function of pixel value.
        
        The functional form is:
            wave = p0(m) + p1(m)*yp + p2(m)*yp**2 + ...)

        with yp = y - y_middle, and:
            p0(m) = q00 + q01 * mp + q02 * mp**2 + ...

        with mp = m_ref/m - 1
        
        This means that the simplest spectrograph model should have:
        q00 : central wavelength or order m_ref
        q01: central wavelength or order m_ref
        q10: central_wavelength/R_pix, with R_pix the resolving power per pixel.
        q11: central_wavelength/R_pix, with R_pix the resolving power per pixel.
        ... with everything else approximately zero.
        
        Parameters
        ----------
        xdeg, ydeg: int
            Order of polynomial
        dir: string
            Directory. If none given, a fit to Tobias's default pixels in "data" is made."""
        if len(init_mod_file)==0:
            params0 = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../data/wavemod.txt'))
        if (len(pixdir) == 0):
            pixdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../data/')
        ms = np.array([])
        waves = np.array([])
        ys = np.array([])
        #The next loop reads in Tobias's wavelengths.
        for m in range(self.m_min, self.m_max+1):
            fname = pixdir + "order{0:d}.txt".format(m)
            pix = np.loadtxt(fname)
            ms = np.append(ms,m*np.ones(pix.shape[0]))
            waves = np.append(waves,pix[:,0])
            #Tobias's definition of the y-axis is backwards compared to python.
            ys = np.append(ys,self.szy - pix[:,1])

        init_resid = self.wave_fit_resid(params0, ms, waves, ys)
        bestp = op.leastsq(self.wave_fit_resid,params0,args=(ms, waves, ys))
        final_resid = self.wave_fit_resid(bestp[0], ms, waves, ys)
        print("Fit residual RMS (Angstroms): {6.3f}".format(np.std(final_resid)))
        params = bestp[0].reshape( (ydeg+1,xdeg+1) )
        outf = open(outdir + "wavemod.txt","w")
        for i in range(ydeg+1):
            for j in range(xdeg+1):
                outf.write("{0:9.4e} ".format(params[i,j]))
            outf.write("\n")
        outf.close()
        
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
        if (self.mode == "single"):
            ## In this case, we obviously ignore the fluxes!
            x = (np.arange(self.im_slit_sz) - self.im_slit_sz/2)*self.microns_pix
            xy = np.meshgrid(x,x)
            ## A simple Gaussian approximation to the fiber far field (or near-field in the
            ## case of the original RHEA2. 2.35482 scales FWHM to stdev
            gsig = self.fib_image_width_in_pix*self.slit_microns_per_det_pix/2.35482
            sim_im = np.exp( - xy[0]**2/2.0/(gsig/self.xbin)**2 - xy[1]**2/2.0/gsig**2  )
        else:
            print("Error: invalid mode " + self.mode)
            raise UserWarning
        return sim_im
        
    def spectral_format(self,xoff=0.0,yoff=0.0,ccd_centre={}):
        """Create a spectrum, with wavelengths sampled in 2 orders.
        
        Parameters
        ----------
        xoff: float
            An input offset from the field center in the slit plane in 
            mm in the x (spatial) direction.
        yoff: float
            An input offset from the field center in the slit plane in
            mm in the y (spectral) direction.
        ccd_centre: dict
            An input describing internal parameters for the angle of the center of the 
            CCD. To run this program multiple times with the same co-ordinate system, 
            take the returned ccd_centre and use it as an input.
            
        Returns
        -------
        x:  (nm, ny) float array
            The x-direction pixel co-ordinate corresponding to each y-pixel and each
            order (m).    
        wave: (nm, ny) float array
            The wavelength co-ordinate corresponding to each y-pixel and each
            order (m).
        blaze: (nm, ny) float array
            The blaze function (pixel flux divided by order center flux) corresponding
            to each y-pixel and each order (m).
        ccd_centre: dict
            NOT YET IMPLEMENTED 
            Parameters of the internal co-ordinate system describing the center of the
            CCD. 
            
        Notes
        -----
        ff = pyfits.getdata('/Users/mireland/data/rhea2/20150721/20150721_flat-001.fits')
        plt.imshow(ff, aspect='auto')
        xy = np.meshgrid(np.arange(35), np.arange(2200))
        plt.plot(x.T + 1375//2-180,xy[1])
        """

        ## Now lets interpolate onto a pixel grid rather than the arbitrary wavelength
        ## grid we began with.
        nm = self.m_max-self.m_min+1
        x_int = np.zeros( (nm,self.szy) )
        wave_int = np.zeros((nm,self.szy) )
        blaze_int = np.zeros((nm,self.szy) )
        
        ## Should be an option here to get files from a different directory.
        wparams = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../data/wavemod.txt'))
        xparams = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../data/xmod.txt'))
        
        ys = np.arange(self.szy)
        ## Loop through m 
        for m in np.arange(self.m_min,self.m_max+1):
            #First, sort out the wavelengths
            mp = self.m_ref/m - 1
            
            #Find the polynomial coefficients for each order.
            ydeg = wparams.shape[0] - 1
            ps = np.empty( (ydeg+1) )
            for i in range(ydeg+1):
                polyq = np.poly1d(wparams[i,:])
                ps[i] = polyq(mp)
            polyp = np.poly1d(ps)      
            wave_int[m - self.m_min,:] = polyp(ys - self.szy/2)
            
            #Find the polynomial coefficients for each order.
            ydeg = xparams.shape[0] - 1
            ps = np.empty( (ydeg+1) )
            for i in range(ydeg+1):
                polyq = np.poly1d(xparams[i,:])
                ps[i] = polyq(mp)
            polyp = np.poly1d(ps)      
            x_int[m - self.m_min,:] = polyp(ys - self.szy/2)
            
            #Finally, the blaze
            wcen = wave_int[m - self.m_min,self.szy/2]
            disp = wave_int[m - self.m_min,self.szy/2+1] - wcen
            order_width = (wcen/m)/disp
            blaze_int[m - self.m_min,:] = np.sinc( (ys-self.szy/2)/order_width)**2

        return x_int,wave_int,blaze_int

    def adjust_x(self, old_x, image, num_xcorr=21):
        """Adjust the x pixel value based on an image and an initial array from 
        spectral_format(). Only really designed for a single fiber flat or science
        image.
        
        Parameters
        ----------
        old_x: numpy array
            An old x pixel array
        image: numpy array
            An image read in from a fits file
        
        Returns
        -------
        A new value of the x array.
        """
        #Create an array with a single pixel with the value 1.0 at the expected peak of
        #each order.
        single_pix_orders = np.zeros(image.shape)
        xy = np.meshgrid(np.arange(old_x.shape[0]), np.arange(old_x.shape[1]))
        single_pix_orders[np.round(xy[1]).astype(int), np.round(old_x.T + self.szx//2).astype(int)] = 1.0
        
        #Make an array of cross-correlation values.
        xcorr = np.zeros(num_xcorr)
        for i in range(num_xcorr):
            xcorr[i] = np.sum(np.roll(single_pix_orders,i-num_xcorr//2,axis=1)*image)
        
        #Based on the maximum cross-correlation, adjust the model x values.
        the_shift = np.argmax(xcorr) - num_xcorr//2

        return old_x+the_shift
        
    def fit_x_to_image(self, image, outdir='./', decrease_dim=10, search_pix=5):
        """Fit a "tramline" map"""
        xx,wave,blaze=self.spectral_format()
        xs = self.adjust_x(xx,image)
#        the_shift = xx_new[0]-xx[0]
        image_med = image.reshape( (image.shape[0]//decrease_dim,decrease_dim,image.shape[1]) )
        image_med = np.median(image_med,axis=1)
        my = np.meshgrid(np.arange(xx.shape[1]), np.arange(xx.shape[0]) + self.m_min)
        ys = my[0]
        ys = np.average(ys.reshape(xs.shape[0], xs.shape[1]//decrease_dim,decrease_dim),axis=2)
        xs = np.average(xs.reshape(xs.shape[0], xs.shape[1]//decrease_dim,decrease_dim),axis=2)
        
        #Now go through and find the peak pixel values. TODO: find a sub-pixel peak.
        for i in range(xs.shape[0]): #Go through each order...
            for j in range(xs.shape[1]):
                xi = int(np.round(xs[i,j]))
                peakpix = image_med[j,self.szx//2 + xi -search_pix:self.szx//2 + xi +search_pix+1]
                xs[i,j] += np.argmax(peakpix) - search_pix
                
        self.fit_to_x(xs,ys=ys)
        
    def fit_to_x(self, x_to_fit, init_mod_file='', outdir='./', ydeg=2, xdeg=4, ys=[], decrease_dim=1):
        """Fit to an (nm,ny) array of x-values.
        
        The functional form is:
            x = p0(m) + p1(m)*yp + p2(m)*yp**2 + ...)

        with yp = y - y_middle, and:
            p0(m) = q00 + q01 * mp + q02 * mp**2 + ...

        with mp = m_ref/m - 1
        
        This means that the simplest spectrograph model should have:
        q00 : central order y pixel
        q01:  spacing between orders divided by the number of orders
        ... with everything else approximately zero.
        
        Parameters
        ----------
        xdeg, ydeg: int
            Order of polynomial
        dir: string
            Directory. If none given, a fit to Tobias's default pixels in "data" is made."""
        if len(init_mod_file)==0:
            params0 = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../data/xmod.txt'))
          
        #Create an array of y and m values.
        xs = x_to_fit.copy()
        my = np.meshgrid(np.arange(xs.shape[1]), np.arange(xs.shape[0]) + self.m_min)
        if (len(ys) == 0):
            ys = my[0]
        ms = my[1]
        
        #Allow a dimensional decrease, for speed
        if (decrease_dim > 1):
            ms = np.average(ms.reshape(xs.shape[0], xs.shape[1]//decrease_dim,decrease_dim),axis=2)
            ys = np.average(ys.reshape(xs.shape[0], xs.shape[1]//decrease_dim,decrease_dim),axis=2)
            xs = np.average(xs.reshape(xs.shape[0], xs.shape[1]//decrease_dim,decrease_dim),axis=2)
        
        #Flatten arrays
        ms = ms.flatten()
        ys = ys.flatten()
        xs = xs.flatten()
        
        #Do the fit!
        init_resid = self.wave_fit_resid(params0, ms, xs, ys,ydeg=ydeg,xdeg=xdeg)
        bestp = op.leastsq(self.wave_fit_resid,params0,args=(ms, xs, ys,ydeg,xdeg))
        final_resid = self.wave_fit_resid(bestp[0], ms, xs, ys,ydeg=ydeg,xdeg=xdeg)
        params = bestp[0].reshape( (ydeg+1,xdeg+1) )

        outf = open(outdir + "xmod.txt","w")
        for i in range(ydeg+1):
            for j in range(xdeg+1):
                outf.write("{0:9.4e} ".format(params[i,j]))
            outf.write("\n")
        outf.close()


    def spectral_format_with_matrix(self):
        """Create a spectral format, including a detector to slit matrix at every point.
        
        Returns
        -------
        x: (nm, ny) float array
            The x-direction pixel co-ordinate corresponding to each y-pixel and each
            order (m).    
        w: (nm, ny) float array
            The wavelength co-ordinate corresponding to each y-pixel and each
            order (m).
        blaze: (nm, ny) float array
            The blaze function (pixel flux divided by order center flux) corresponding
            to each y-pixel and each order (m).
        matrices: (nm, ny, 2, 2) float array
            2x2 slit rotation matrices, mapping output co-ordinates back to the slit.
        """        
        x,w,b = self.spectral_format()
        matrices = np.zeros( (x.shape[0],x.shape[1],2,2) )
        amat = np.zeros((2,2))

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                ## Create a matrix where we map input angles to output coordinates.
                amat[0,0] = 1.0/self.slit_microns_per_det_pix
                amat[0,1] = 0
                amat[1,0] = 0
                amat[1,1] = 1.0/self.slit_microns_per_det_pix
                ## Apply an additional rotation matrix. If the simulation was complete,
                ## this wouldn't be required.
                r_rad = np.radians(self.extra_rot)
                dy_frac = (j - x.shape[1]/2.0)/(x.shape[1]/2.0)
                extra_rot_mat = np.array([[np.cos(r_rad*dy_frac),np.sin(r_rad*dy_frac)],[-np.sin(r_rad*dy_frac),np.cos(r_rad*dy_frac)]])
                amat = np.dot(extra_rot_mat,amat)
                ## We actually want the inverse of this (mapping output coordinates back
                ## onto the slit.
                matrices[i,j,:,:] =  np.linalg.inv(amat)
        return x,w,b,matrices
                
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
