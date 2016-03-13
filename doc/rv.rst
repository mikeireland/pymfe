Radial Velocity Algorithms
==========================

The goal of the radial velocity code in pymfe is to be able to take into account variable pixel-based uncertainties (e.g. readout noise versus photon noise, and bad pixels), and to eventually to enable a non-Gaussian error model. Together, these features should enable an optimal data extraction. This general list of data reduction steps is given here because it probably doesn't all belong in the RV module!

The steps in the data reduction are:

1) Fitting the spectrum tracks :math:`y_m(x)` to the flat (white light) data for orders m and dispersion-direction pixels x. See the instrument pages (e.g. rhea) for these details.

2) Fitting the wavelength scale :math:`\lambda_m(x)` to the arc lines. See the instrument pages (e.g. rhea) for these details.

3) Dark correcting the data. Currently done in the RV module in RadialVelocity.extract_spectra.

4) Extracting the data and the flat fields, to form :math:`f_m(x)`, the flux for orders m and dispersion direction pixels x. See the extract module for these details (called from RadialVelocity.extract_spectra).

5) Normalising the flat fields, so that the median of each order is 1.0 (in RadialVelocity.extract_spectra).

6) Dividing by the extracted flat field. Uncertainties from the flat field are added in quadrature (in RadialVelocity.extract_spectra).

7) Optionally, creating a reference spectrum from the data themselves. This is the RadialVelocity.create_ref_spect method.

8) Fitting a shifted and normalised reference spectrum to each spectrum, giving a radial velocity shift. This is the RadialVelocity.calculate_rv_shift method.

9) Averaging the radial velocities, and correcting for Th/Ar shifts (currently not in this module - in a script combine_rvs.py).

The key routines are described in their docstrings.


The rv module
==================

.. automodule:: pymfe.rv
    :members:

