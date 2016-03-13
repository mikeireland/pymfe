Extraction Algorithms
=====================

The extraction algorithms are based on the mathematics in Sharp and Birchall (2009) 
(optimal extraction) and in Bolton and Schlegel (2009) (spectro-perfectionism). Neither
algorithm is used verbatim, because of the unique data analysis challenge of a long
fiber slit for GHOST that does not necessarily have adequate sampling. Instead, each
wavelength in each order (corresponding a single spectral-direction pixel in the center
of the slit image) has a nominal floating-point dispersion direction pixel coordinate
assigned to it. Then each row (i.e. dispersion direction) is multipled with a function
that is non-zero on up to 2 pixels, with a centroid equal to the pixel co-ordinate for that wavelength. Each row is then summed (i.e. the multiplication and sum is like a 
convolution) and the resulting column is treated exactly the same as in Sharp and Birchall
2009. 

Currently (July 2015), the sky and star extract very well, with equivalent widths and
resolution consistent between 1D and 2D extraction. However, the Th/Xe fiber gives the
same flux in 1D and 2D extraction for the flat lamp, but quite different fluxes whenever
there is a slit tilt for the arc lines. There is also some ringing in a Th/Xe fiber 
flat at extracted pixel
numbers higher than 3000 (where there is row tilt due to curvature), and a 
little deconvolution ringing evident in the Th/Xe 
2D extraction (also present, but to a lesser extent, in the 1D extraction). There 
may to be a ~0.1 pixel error in the extraction profile, but no more than that. So these
Th/Xe issues definitely seem to be an extraction artefact...

The extract module
==================

Note - only the Extractor class is currently imported,
so only that should be documented!

.. automodule:: pymfe.extract
    :members:

