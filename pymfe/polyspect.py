"""This is a simple simulation and extraction definition code for RHEA.
The key is to use Tobias's spectral fitting parameters and define
the same kinds of extraction arrays as pyghost.
"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button, RadioButtons
import optics
import os
import scipy.optimize as op
import pdb
import astropy.io.fits as pyfits


class Polyspect(object):
    """A class containing tools common for any spectrograph.
    This class should be inhereted by the specific spectrograph module.
    Contains functions related to polynomial modelling of spectrograph orders.
    """

    def wave_fit_resid(self, params, ms, waves, ys, ydeg=3, xdeg=3):
        """A fit function for read_lines_and_fit (see that function for details)
        to be used in scipy.optimize.leastsq. The same function is used in
        fit_to_x, but in that case "waves" is replaced by "xs".
        """
        if np.prod(params.shape) != (xdeg + 1) * (ydeg + 1):
            print("Parameters are flattened - xdeg and ydeg must be correct!")
            raise UserWarning
        params = params.reshape((ydeg + 1, xdeg + 1))
        if (len(ms) != len(ys)):
            print("ms and ys must all be the same length!")
            raise UserWarning
        mp = self.m_ref / ms - 1

        ps = np.empty((len(ms), ydeg + 1))
        # Find the polynomial coefficients for each order.
        for i in range(ydeg + 1):
            polyq = np.poly1d(params[i, :])
            ps[:, i] = polyq(mp)
        wave_mod = np.empty(len(ms))
        for i in range(len(ms)):
            polyp = np.poly1d(ps[i, :])
            wave_mod[i] = polyp(ys[i] - self.szy // 2)
        return wave_mod - waves

    def read_lines_and_fit(self, init_mod_file='', pixdir='',
                           outdir='./', ydeg=3, xdeg=3, residfile='resid.txt'):
        """Read in a series of text files that have a (Wavelength, pixel)
        format in file names like order99.txt and order100.txt.
        Fit an nth order polynomial to the wavelength as a function
        of pixel value.

        The functional form is:
            wave = p0(m) + p1(m)*yp + p2(m)*yp**2 + ...)

        with yp = y - y_middle, and:
            p0(m) = q00 + q01 * mp + q02 * mp**2 + ...

        with mp = m_ref/m - 1

        This means that the simplest spectrograph model should have:
        q00 : central wavelength or order m_ref
        q01: central wavelength or order m_ref
        q10: central_wavelength/R_pix, with R_pix the resolving power / pixel.
        q11: central_wavelength/R_pix, with R_pix the resolving power / pixel.
        ... with everything else approximately zero.

        Parameters
        ----------
        xdeg, ydeg: int
            Order of polynomial
        dir: string
            Directory. If none given, a fit to Tobias's default
            pixels in "data" is made."""
        if len(init_mod_file) == 0:
            params0 = np.loadtxt(os.path.join(os.path.dirname(
                os.path.abspath(__file__)), '../data/' +
                self.spect + '/wavemod.txt'))
        if (len(pixdir) == 0):
            pixdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '../data/' + self.spect + '/')
        # The next loop reads in Mike's or Tobias's wavelengths.
        # Try for Mike's single file format first.
        # To make this neater, it could be a function that overrides this
        # base class.
        if os.path.exists(pixdir + "arclines.txt"):
            lines = np.loadtxt(pixdir + "arclines.txt")
            ms = lines[:, 3]
            waves = lines[:, 0]
            ys = lines[:, 1]
        else:
            ms = np.array([])
            waves = np.array([])
            ys = np.array([])
            for m in range(self.m_min, self.m_max + 1):
                fname = pixdir + "order{0:d}.txt".format(m)
                try:
                    pix = np.loadtxt(fname)
                except:
                    print("Error: arc line files don't exist!")
                    raise UserWarning
                ms = np.append(ms, m * np.ones(pix.shape[0]))
                waves = np.append(waves, pix[:, 0])
                # Tobias's definition of the y-axis is backwards compared to
                # python.
                ys = np.append(ys, self.szy - pix[:, 1])

        init_resid = self.wave_fit_resid(params0, ms, waves, ys, ydeg=ydeg,
                                         xdeg=xdeg)
        bestp = op.leastsq(self.wave_fit_resid, params0, args=(ms, waves, ys,
                                                               ydeg, xdeg))
        final_resid = self.wave_fit_resid(bestp[0], ms, waves, ys, ydeg=ydeg,
                                          xdeg=xdeg)
        # Output the fit residuals.
        wave_and_resid = np.array([waves, ms, final_resid]).T
        np.savetxt(outdir + residfile, wave_and_resid, fmt='%9.4f %d %7.4f')
        print("Fit residual RMS (Angstroms): {0:6.3f}".format(
            np.std(final_resid)))
        params = bestp[0].reshape((ydeg + 1, xdeg + 1))
        outf = open(outdir + "wavemod.txt", "w")
        for i in range(ydeg + 1):
            for j in range(xdeg + 1):
                outf.write("{0:10.5e} ".format(params[i, j]))
            outf.write("\n")
        outf.close()

    def spectral_format(self, xoff=0.0, yoff=0.0, ccd_centre={}, wparams=None,
                        xparams=None, imgfile=None):
        """Create a spectrum, with wavelengths sampled in 2 orders based on
           a pre-existing wavelength and x position polynomial model.

        Parameters
        ----------
        xoff: float
            An input offset from the field center in the slit plane in
            mm in the x (spatial) direction.
        yoff: float
            An input offset from the field center in the slit plane in
            mm in the y (spectral) direction.
        ccd_centre: dict
            An input describing internal parameters for the angle of
            the center of the CCD. To run this program multiple times
            with the same co-ordinate system, take the returned
            ccd_centre and use it as an input.
        imgfile: string (optional)
            String containing a file name for an image. This function
            uses this image and over plots the created spectrum.

        Returns
        -------
        x:  (nm, ny) float array
            The x-direction pixel co-ordinate corresponding to each y-pixel and
            each order (m).
        wave: (nm, ny) float array
            The wavelength co-ordinate corresponding to each y-pixel and each
            order (m).
        blaze: (nm, ny) float array
            The blaze function (pixel flux divided by order center flux)
            corresponding to each y-pixel and each order (m).
        ccd_centre: dict
            NOT YET IMPLEMENTED
            Parameters of the internal co-ordinate system describing the
            center of the CCD.


        """

        # Now lets interpolate onto a pixel grid rather than the arbitrary
        # wavelength grid we began with.
        nm = self.m_max - self.m_min + 1
        x_int = np.zeros((nm, self.szy))
        wave_int = np.zeros((nm, self.szy))
        blaze_int = np.zeros((nm, self.szy))

        # Should be an option here to get files from a different directory.
        if wparams is None:
            wparams = np.loadtxt(os.path.join(os.path.dirname(
                os.path.abspath(__file__)), '../data/' +
                self.spect + '/wavemod.txt'))
        if xparams is None:
            xparams = np.loadtxt(os.path.join(os.path.dirname(
                os.path.abspath(__file__)), '../data/' + self.spect +
                                              '/xmod.txt'))

        ys = np.arange(self.szy)
        # Loop through m
        for m in np.arange(self.m_min, self.m_max + 1):
            # First, sort out the wavelengths
            mp = self.m_ref / m - 1

            # Find the polynomial coefficients for each order.
            ydeg = wparams.shape[0] - 1
            ps = np.empty((ydeg + 1))
            for i in range(ydeg + 1):
                polyq = np.poly1d(wparams[i, :])
                ps[i] = polyq(mp)
            polyp = np.poly1d(ps)
            wave_int[m - self.m_min, :] = polyp(ys - self.szy // 2)

            # Find the polynomial coefficients for each order.
            ydeg = xparams.shape[0] - 1
            ps = np.empty((ydeg + 1))
            for i in range(ydeg + 1):
                polyq = np.poly1d(xparams[i, :])
                ps[i] = polyq(mp)
            polyp = np.poly1d(ps)
            x_int[m - self.m_min, :] = polyp(ys - self.szy // 2)

            # Finally, the blaze
            wcen = wave_int[m - self.m_min, self.szy / 2]
            disp = wave_int[m - self.m_min, self.szy / 2 + 1] - wcen
            order_width = (wcen / m) / disp
            blaze_int[m - self.m_min, :] = np.sinc((ys - self.szy / 2) /
                                                   order_width)**2

        # Plot this if we have an image file
        if imgfile:
            im = pyfits.getdata(imgfile)
            if not self.transpose:
                im = im.T
            plt.clf()
            plt.imshow(np.arcsinh((im - np.median(im)) / 100), aspect='auto',
                       interpolation='nearest', cmap=cm.gray)
            plt.axis([0, im.shape[1], im.shape[0], 0])
            plt.plot(x_int.T + + self.szx // 2)

        return x_int, wave_int, blaze_int

    def adjust_x(self, old_x, image, num_xcorr=21):
        """Adjust the x pixel value based on an image and an initial
            array from spectral_format(). Only really designed for a
            single fiber flat or science image. This is a helper
            routine for fit_x_to_image.

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
        # Create an array with a single pixel with the value 1.0 at the
        # expected peak of each order.
        single_pix_orders = np.zeros(image.shape)
        xy = np.meshgrid(np.arange(old_x.shape[0]), np.arange(old_x.shape[1]))
        single_pix_orders[np.round(xy[1]).astype(int),
                          np.round(old_x.T + self.szx // 2).astype(int)] = 1.0

        # Make an array of cross-correlation values.
        xcorr = np.zeros(num_xcorr)
        for i in range(num_xcorr):
            xcorr[i] = np.sum(np.roll(single_pix_orders, i - num_xcorr // 2,
                                      axis=1) * image)

        # Based on the maximum cross-correlation, adjust the model x values.
        the_shift = np.argmax(xcorr) - num_xcorr // 2

        return old_x + the_shift

    def fit_x_to_image(self, image, decrease_dim=10, search_pix=20, xdeg=4):
        """Fit a "tramline" map. Note that an initial map has to be pretty
        close, i.e. within "search_pix" everywhere. To get within search_pix
        everywhere, a simple model with a few paramers is fitted manually.
        This can be done with a GUI using the slider_adjust function.

        Parameters
        ----------
        image: numpy array
            The image of a single reference fiber to fit to.
        decrease_dim: int
            Median filter by this amount in the dispersion direction and
            decrease the dimensionality of the problem accordingly.
            This helps with both speed and robustness.
        search_pix: int
            Search within this many pixels of the initial model.
        """
        xx, wave, blaze = self.spectral_format()
        xs = self.adjust_x(xx, image)

        # Median-filter in the dispersion direction.
        image_med = image.reshape((image.shape[0] // decrease_dim,
                                   decrease_dim, image.shape[1]))
        image_med = np.median(image_med, axis=1)
        my = np.meshgrid(np.arange(xx.shape[1]), np.arange(xx.shape[0]) +
                         self.m_min)
        ys = my[0]
        ys = np.average(ys.reshape(xs.shape[0], xs.shape[1] // decrease_dim,
                                   decrease_dim), axis=2)
        xs = np.average(xs.reshape(xs.shape[0], xs.shape[1] // decrease_dim,
                                   decrease_dim), axis=2)

        # Now go through and find the peak pixel values.
        for i in range(xs.shape[0]):  # Go through each order...
            for j in range(xs.shape[1]):
                xi = int(np.round(xs[i, j]))
                peakpix = image_med[j, self.szx // 2 + xi -
                                    search_pix:self.szx // 2 +
                                    xi + search_pix + 1]
                xs[i, j] += np.argmax(peakpix) - search_pix

        self.fit_to_x(xs, ys=ys, xdeg=xdeg)

    def fit_to_x(self, x_to_fit, init_mod_file='', outdir='./', ydeg=2,
                 xdeg=4, ys=[], decrease_dim=1):
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
        """
        if len(init_mod_file) == 0:
            params0 = np.loadtxt(os.path.join(os.path.dirname(
                os.path.abspath(__file__)), '../data/' + self.spect +
                                              '/xmod.txt'))

        # Create an array of y and m values.
        xs = x_to_fit.copy()
        my = np.meshgrid(np.arange(xs.shape[1]), np.arange(xs.shape[0]) +
                         self.m_min)
        if (len(ys) == 0):
            ys = my[0]
        ms = my[1]

        # Allow a dimensional decrease, for speed
        if (decrease_dim > 1):
            ms = np.average(ms.reshape(xs.shape[0], xs.shape[
                            1] // decrease_dim, decrease_dim), axis=2)
            ys = np.average(ys.reshape(xs.shape[0], xs.shape[1] //
                                       decrease_dim, decrease_dim), axis=2)
            xs = np.average(xs.reshape(xs.shape[0], xs.shape[1] //
                                       decrease_dim, decrease_dim), axis=2)

        # Flatten arrays
        ms = ms.flatten()
        ys = ys.flatten()
        xs = xs.flatten()

        # Do the fit!
        init_resid = self.wave_fit_resid(
            params0, ms, xs, ys, ydeg=ydeg, xdeg=xdeg)
        bestp = op.leastsq(self.wave_fit_resid, params0,
                           args=(ms, xs, ys, ydeg, xdeg))
        final_resid = self.wave_fit_resid(
            bestp[0], ms, xs, ys, ydeg=ydeg, xdeg=xdeg)
        params = bestp[0].reshape((ydeg + 1, xdeg + 1))
        print(init_resid, final_resid)
        outf = open(outdir + "xmod.txt", "w")
        for i in range(ydeg + 1):
            for j in range(xdeg + 1):
                outf.write("{0:9.4e} ".format(params[i, j]))
            outf.write("\n")
        outf.close()

    def spectral_format_with_matrix(self):
        """Create a spectral format, including a detector to slit matrix at
           every point.

        Returns
        -------
        x: (nm, ny) float array
            The x-direction pixel co-ordinate corresponding to each y-pixel
            and each order (m).
        w: (nm, ny) float array
            The wavelength co-ordinate corresponding to each y-pixel and each
            order (m).
        blaze: (nm, ny) float array
            The blaze function (pixel flux divided by order center flux)
            corresponding to each y-pixel and each order (m).
        matrices: (nm, ny, 2, 2) float array
            2x2 slit rotation matrices, mapping output co-ordinates back
            to the slit.
        """
        x, w, b = self.spectral_format()
        matrices = np.zeros((x.shape[0], x.shape[1], 2, 2))
        amat = np.zeros((2, 2))

        for i in range(x.shape[0]):  # i is the order
            for j in range(x.shape[1]):
                # Create a matrix where we map input angles to output
                # coordinates.
                slit_microns_per_det_pix = self.slit_microns_per_det_pix_first +
                float(i) / x.shape[0] * (self.slit_microns_per_det_pix_last -
                                         self.slit_microns_per_det_pix_first)
                amat[0, 0] = 1.0 / slit_microns_per_det_pix
                amat[0, 1] = 0
                amat[1, 0] = 0
                amat[1, 1] = 1.0 / slit_microns_per_det_pix
                # Apply an additional rotation matrix. If the simulation was
                # complete, this wouldn't be required.
                r_rad = np.radians(self.extra_rot)
                dy_frac = (j - x.shape[1] / 2.0) / (x.shape[1] / 2.0)
                extra_rot_mat = np.array([[np.cos(r_rad * dy_frac),
                                           np.sin(r_rad * dy_frac)],
                                          [-np.sin(r_rad * dy_frac),
                                           np.cos(r_rad * dy_frac)]])
                amat = np.dot(extra_rot_mat, amat)
                # We actually want the inverse of this (mapping output
                # coordinates back onto the slit.
                matrices[i, j, :, :] = np.linalg.inv(amat)
        return x, w, b, matrices

    def slit_flat_convolve(self, flatfield=None, mode='ghoststd'):
        """Function that takes a flat field image and a slit profile and
           convolves the two in 2D. Returns result of convolution, which
           should be used for tramline fitting. ONLY WORKS FOR GHOST.

        Parameters
        ----------
        flatimage: string
            A file name containing a flat field image from the spectrograph
        mode: string
            Either 'ghoststd' of 'ghosthigh'. This will inform on the order
            profile.

        Returns
        -------
        flat_conv: float array
            Currently returns the convolved 2D array.

        """
        # Grab the flat field data
        flat = pyfits.getdata(flatfield)

        # At this point create a slit profile
        # Create a x baseline for convolution and assume a FWHM for profile
        x = flat.shape[0]
        profilex = np.arange(x) - x // 2
        sigma = 1.1
        # This is the fiber centre separation in pixels. MAY NEED TO BE
        # ADJUSTED DEPENDING ON MODE!
        fiber_separation = 3.97
        # Now create a model of the slit profile
        mod_slit = np.zeros(x)
        if self.mode == 'high':
            nfibers = 26
        else:
            nfibers = self.nl

        for i in range(-nfibers // 2, nfibers // 2):
            mod_slit += np.exp(-(profilex - i * fiber_separation)**2 /
                               2.0 / sigma**2)

        # Normalise the slit model and fourier transform for convolution
        mod_slit /= np.sum(mod_slit)
        mod_slit_ft = np.fft.rfft(np.fft.fftshift(mod_slit))
        # Fourier the flat for convolution
        im_fft = np.fft.rfft(flat, axis=0)
        flat_conv = np.zeros_like(im_fft)
        # Now convolved in 2D
        for i in range(im_fft.shape[1]):
            flat_conv[:, i] = im_fft[:, i] * mod_slit_ft
        flat_conv = np.fft.irfft(flat_conv, axis=0)
        return flat_conv

    def slider_adjust(self, datafile, wparams='./data/ghost/wavemod.txt',
                      xparams='./data/ghost/xmod.txt',
                      percentage_variation=10):
        """Function that uses matplotlib slider widgets to adjust a polynomial
        model overlaid on top of a flat field image. In practice this will be
        overlaid on top of the result of convolving the flat with a slit
        profile in 2D which just reveals the location of the middle of the
        orders.

        Parameters
        ----------
        datafile: string
            A string containing the location of the data file (flat field)
            to be used as a visual comparison of the model
        wparams: string
            location of the wavemod.txt file containing the initial wavelength
            model parameters.
        xparams: string
            location of the xmod.txt file containing the initial order location
            model parameters.
        percentage_variation: int
            How much should the percentage adjustment in each bin as a function
            of the parameter. Default is 10%

        Returns
        -------
        Either a success or an error message.

        """

        # Grab the data file
        data = pyfits.getdata(datafile)
        nx = data.shape[0]
        ghost = pymfe.ghost.Arm('blue', mode='std')
        wparams = np.loadtxt('./data/ghost/wavemod.txt')
        xparams = np.loadtxt('./data/ghost/xmod.txt')

        fig, ax = plt.subplots()

        x_int, wave_int, blaze_int = ghost.spectral_format(wparams=wparams,
                                                           xparams=xparams)
        y = np.meshgrid(np.arange(data.shape[1]), np.arange(x_int.shape[0]))[0]

        l, = plt.plot(y.flatten()[::10], x_int.flatten()[::10] + nx // 2,
                      color='green', linestyle='None', marker='.')

        flat_conv = ghost.slit_flat_convolve(flatfield=datafile)
        ax.imshow((flat_conv - np.median(flat_conv)) / 1e2)

        axcolor = 'lightgoldenrodyellow'

        # Create a second window for cliders.
        slide_fig = plt.figure()

        # This function is executed on each slider change.
        # spectral_format is updated.
        def update(val):
            for i in range(npolys):
                for j in range(polyorder):
                    xparams[i, j] = sliders[i][j].val
            x, wave, blaze = ghost.spectral_format(wparams=wparams,
                                                   xparams=xparams)
            l.set_ydata(x.flatten()[::10] + nx // 2)
            fig.canvas.draw_idle()

        polyorder = xparams.shape[1]
        npolys = xparams.shape[0]
        # Now we start putting sliders in depending on number of parameters
        height = 1. / (npolys * 2)
        width = 1. / (polyorder * 2)
        # Use this to adjust in a percentage how much to let each parameter
        # vary
        frac_xparams = np.absolute(xparams * (percentage_variation / 100))
        axq = [[0 for x in range(polyorder)] for y in range(npolys)]
        sliders = [[0 for x in range(polyorder)] for y in range(npolys)]
        # Now put all the sliders in the new figure based on position in the
        # array
        for i in range(npolys):
            for j in range(polyorder):
                left = j * width * 2
                bottom = 1 - (i + 1) * height * 2 + height
                axq[i][j] = plt.axes([left, bottom, width, height],
                                     axisbg=axcolor)
                if xparams[i, j] == 0:
                    sliders[i][j] = Slider(axq[i][j],
                                           'test' + str(i) + str(j), 0, 0.1,
                                           valinit=xparams[i, j])
                else:
                    sliders[i][j] = Slider(axq[i][j], 'test' + str(i) + str(j),
                                           xparams[i, j] - frac_xparams[i, j],
                                           xparams[i, j] + frac_xparams[i, j],
                                           valinit=xparams[i, j])
                plt.legend(loc=3)
                sliders[i][j].on_changed(update)

        # Have a button to output the current varied model to the correct file
        submitax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(submitax, 'Submit', color=axcolor, hovercolor='0.975')

        def submit(event):
            try:
                np.savetxt('data/ghost/xmod.txt', xparams, fmt='%.4e')
                print('Data updated on xmod.txt')
                return 1
            except Exception:
                return 'Unable to save data onto file.'

        button.on_clicked(submit)

        """THINGS TO ADD:
        -All outputs of each procedure must be fits files, including the
        polynomial fit parameters.
        -Multiple files for each variation of the spectrograph
        -A fit button?
        """

        plt.show()
        return 1
