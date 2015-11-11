# pymfe
Python-based analysis for multiple fiber échelle spectroscopy.

[This is **in-progress**, and only on github in order to collaborate prior to it really working]

There is a long history of échelle reduction software in astronomy, both for slit- and fiber-fed spectrographs. When a spectrograph is fed by mulitple fibers, the analysis changes somewhat, due to e.g. cross-talk between neighboring fibers. In the case of all fibers observing the same object (e.g. Veloce, GHOST, RHEA) and a goal of radial velocity, a forward-modelling (*) approach is ideal, and quite different to existing pipelines. For these reasons, a new pipeline has been created here.

For other types of data, pipelines worth looking into include:
- IRAF: A gold standard in manual, slit-fed spectroscopy analysis. Written largely in Fortran, with a python wrapper.
- OPERA: A fiber-fed analysis package (in C++) for Espadons.
- ...?

TODO:
1) Change "optics" to "opticstools"

(*) By "forward-modelling" we mean a pixel-based convolution of a reference spectrum by a slit image, fitting for the radial-velocity (i.e. sub-pixel shifts).
