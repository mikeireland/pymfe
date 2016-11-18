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

    def __init__(self, arm='blue', mode='std'):
        """Initialisation function that sets all the mode specific parameters
        related to each configuration of the spectrograph. 
        """
        self.spect = 'ghost'
        self.arm = arm
        self.d = 1000 / 52.67  # Distance in microns
        self.theta = 65.0  # Blaze angle
        self.assym = 1.0 / 0.41  # Magnification
        self.gamma = 0.56  # Echelle gamma
        self.nwave = 1e2  # Wavelengths per order for interpolation.
        self.f_col = 1750.6  # Collimator focal length.
        self.lenslet_high_size = 118.0  # Lenslet flat-to-flat in microns
        self.lenslet_std_size = 197.0  # Lenslet flat-to-flat in microns
        self.microns_pix = 2.0  # When simulating the slit image, use this many microns per pixel
        self.microns_arcsec = 400.0  # Number of microns in the slit image plane per arcsec
        self.im_slit_sz = 2048  # Size of the image slit size in pixels.
        if (arm == 'red'):
            # Additional slit rotation across an order needed to match Zemax.
            self.extra_rot = 3.0
            self.szx = 6144
            self.szy = 6144
            self.f_cam = 264.0
            self.px_sz = 15e-3
            self.drot = -2.0  # Detector rotation
            self.d_x = 1000 / 565.  # VPH line spacing
            self.theta_i = 30.0  # Prism incidence angle
            self.alpha1 = 0.0  # First prism apex angle
            self.alpha2 = 0.0  # Second prism apex angle
            self.m_min = 34
            self.m_max = 67
            self.m_ref = 50  # Reference order
        elif (arm == 'blue'):
            # Additional slit rotation accross an order needed to match Zemax.
            self.extra_rot = 2.0
            self.szx = 4096
            self.szy = 4112
            self.f_cam = 264.0
            self.px_sz = 15e-3
            self.d_x = 1000 / 1137.  # VPH line spacing
            self.theta_i = 30.0  # Prism incidence angle
            self.drot = -2.0  # Detector rotation.
            self.alpha1 = 0.0  # First prism apex angle
            self.alpha2 = 0.0  # Second prism apex angle
            self.m_min = 63
            self.m_max = 95
            self.m_ref = 80  # Reference order
        else:
            print("Unknown spectrograph arm!")
            raise UserWarning

        if (mode == 'high'):
            self.mode = mode
            self.lenslet_width = self.lenslet_high_size
            self.nl = 28
            # Set default profiles - object, sky and reference
            fluxes = np.zeros((self.nl, 3))
            fluxes[2:21, 0] = 0.37
            fluxes[8:15, 0] = 0.78
            fluxes[11, 0] = 1.0
            # NB if on the following line, fluxes[2:,1]=1.0 is set, sky will be
            # subtracted automatically.
            fluxes[2 + 19:, 1] = 1.0
            fluxes[0, 2] = 1.0
        elif (mode == 'std'):
            self.mode = mode
            self.lenslet_width = self.lenslet_std_size
            self.nl = 17
            # Set default profiles - object 1, sky and object 2
            fluxes = np.zeros((self.nl, 3))
            fluxes[0:7, 0] = 1.0
            fluxes[7:10, 1] = 1.0
            fluxes[10:, 2] = 1.0
        else:
            print("Unknown mode!")
            raise UserWarning

    def make_lenslets(self, fluxes=[], seeing=0.8, llet_offset=0):
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
        s32 = np.sqrt(3) / 2
        hex_scale = 1.15
        conv_fwhm = 30.0  # equivalent to a 1 degree FWHM for an f/3 input ??? !!! Double-check !!!

        if self.mode == 'std':
            yoffset = (self.lenslet_width / self.microns_pix / hex_scale *
                       np.array([0, -s32, s32, 0, -s32, s32, 0])).astype(int)
            xoffset = (self.lenslet_width / self.microns_pix / hex_scale *
                       np.array([-1, -0.5, -0.5, 0, 0.5, 0.5, 1.0])).astype(int)
        elif self.mode == 'high':
            yoffset = (self.lenslet_width / self.microns_pix / hex_scale * s32 * np.array(
                [-2, 2, -2, -1, -1, 0, -1, -1, 0, 0, 0, 1, 1, 0, 1, 1, 2, -2, 2])).astype(int)
            xoffset = (self.lenslet_width / self.microns_pix / hex_scale * 0.5 * np.array(
                [-2, 0, 2, -3, 3, -4, -1, 1, -2, 0, 2, -1, 1, 4, -3, 3, -2, 0, 2])).astype(int)
        else:
            print("Error: mode must be standard or high")

        # Some preliminaries...
        cutout_hw = int(lenslet_width / self.microns_pix * 1.5)
        im_slit = np.zeros((szy, szx))
        x = np.arange(szx) - szx / 2.0
        y = np.arange(szy) - szy / 2.0
        xy = np.meshgrid(x, y)
        # r and wr enable the radius from the lenslet center to be indexed
        r = np.sqrt(xy[0]**2 + xy[1]**2)
        wr = np.where(r < 2 * lenslet_width / self.microns_pix)
        # g is a Gaussian used for FRD
        g = np.exp(-r**2 / 2.0 / (conv_fwhm / self.microns_pix / 2.35)**2)
        g = np.fft.fftshift(g)
        g /= np.sum(g)
        gft = np.conj(np.fft.rfft2(g))
        pix_size_slit = self.px_sz * \
            (self.f_col / self.assym) / self.f_cam * 1000.0 / self.microns_pix
        pix = np.zeros((szy, szx))
        pix[np.where((np.abs(xy[0]) < pix_size_slit / 2) *
                     (np.abs(xy[1]) < pix_size_slit / 2))] = 1
        pix = np.fft.fftshift(pix)
        pix /= np.sum(pix)
        pix_ft = np.conj(np.fft.rfft2(pix))
        # Create some hexagons. We go via a "cutout" for efficiency.
        h_cutout = optics.hexagon(
            szy, lenslet_width / self.microns_pix * fillfact / hex_scale)
        hbig_cutout = optics.hexagon(
            szy, lenslet_width / self.microns_pix * fillfact)
        h = np.zeros((szy, szx))
        hbig = np.zeros((szy, szx))
        h[:, szx / 2 - szy / 2:szx / 2 + szy / 2] = h_cutout
        hbig[:, szx / 2 - szy / 2:szx / 2 + szy / 2] = hbig_cutout
        if len(fluxes) != 0:
            # If we're not simulating seeing, the image-plane is uniform, and we only use
            # the values of "fluxes" to scale the lenslet fluxes.
            im = np.ones((szy, szx))
            # Set the offsets to zero because we may be simulating e.g. a single Th/Ar lenslet
            # and not starlight (from the default xoffset etc)
            xoffset = np.zeros(len(fluxes), dtype=int)
            yoffset = np.zeros(len(fluxes), dtype=int)
        else:
            # If we're simulating seeing, create a Moffat function as our input profile,
            # but just make the lenslet fluxes uniform.
            im = np.zeros((szy, szx))
            im_cutout = optics.moffat2d(
                szy, seeing * self.microns_arcsec / self.microns_pix / 2, beta=4.0)
            im[:, szx / 2 - szy / 2:szx / 2 + szy / 2] = im_cutout
            fluxes = np.ones(len(xoffset))

        # Go through the flux vector and fill in each lenslet.
        for i in range(len(fluxes)):
            im_one = np.zeros((szy, szx))
            im_cutout = np.roll(
                np.roll(im, yoffset[i], axis=0), xoffset[i], axis=1) * h
            im_cutout = im_cutout[szy / 2 - cutout_hw:szy / 2 +
                                  cutout_hw, szx / 2 - cutout_hw:szx / 2 + cutout_hw]
            prof = optics.azimuthalAverage(
                im_cutout, returnradii=True, binsize=1)
            prof = (prof[0], prof[1] * fluxes[i])
            xprof = np.append(np.append(0, prof[0]), np.max(prof[0]) * 2)
            yprof = np.append(np.append(prof[1][0], prof[1]), 0)
            im_one[wr] = np.interp(r[wr], xprof, yprof)
            im_one = np.fft.irfft2(np.fft.rfft2(im_one) * gft) * hbig
            im_one = np.fft.irfft2(np.fft.rfft2(im_one) * pix_ft)
            #!!! The line below could add tilt offsets... important for PRV simulation !!!
            #im_one = np.roll(np.roll(im_one, tilt_offsets[0,i], axis=1),tilt_offsets[1,i], axis=0)*hbig
            the_shift = int((llet_offset + i - self.nl / 2.0)
                            * lenslet_width / self.microns_pix)
            im_slit += np.roll(im_one, the_shift, axis=1)
        return im_slit
