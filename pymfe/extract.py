"""Given (x,wave,matrices, slit_profile), extract the flux from each order. For 
readability, we keep this separate from the simulator.... but the simulator is
required in order to run this.

To run, create a simulated fits file (e.g. "test_blue.fits") using ghost module then:

blue_high = pymfe.Extractor(pymfe.ghost.Arm('blue', 'high'))

flux,var = blue_high.two_d_extract("test_blue.fits")

plt.plot(blue_high.w_map[0,:], flux[0,:,0])

"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
try: 
    import pyfits
except:
    import astropy.io.fits as pyfits
import pdb
from astropy.modeling import models, fitting
import matplotlib.cm as cm

class Extractor():
    """A class for each arm of the spectrograph. The initialisation function takes a 
    single string representing the configuration. For GHOST, it can be "red" or "blue".
    
    The extraction is defined by 3 key parameters: an "x_map", which is equivalent to
    2dFDR's tramlines and contains a physical x-coordinate for every y (dispersion direction)
    coordinate and order, and a "w_map", which is the wavelength corresponding to every y
    (dispersion direction) coordinate and order. 
    
    sim must include:
    
    spectral_format_with_matrix()
    make_lenslets(fluxes)
    
    fluxes (nlenslets x nobj) array
    nl (nlenslets)
    szx (size in x [non-dispersion] direction)
    mode (string,for error messages)
    lenslet_width, im_slit_sz, microns_pix (together define the make_lenslets output)
    """
    
    def __init__(self,sim,transpose_data=True,badpixmask=[]):
        self.sim = sim
        self.transpose_data=transpose_data
        self.badpixmask = badpixmask
        self.x_map,self.w_map,self.blaze,self.matrices = self.sim.spectral_format_with_matrix()
        #Fill in the slit dimensions in "simulator pixel"s. based on if we are in the 
        #high or standard resolution mode.
        self.define_profile(sim.fluxes)
            
        #Set some default pixel offsets for each lenslet, as used for a square lenslet profile
        ny = self.x_map.shape[1]
        nm = self.x_map.shape[0]
        pix_offset_ix = np.append(np.append([0],np.arange(1,sim.nl).repeat(2)),sim.nl)
        self.square_offsets = np.empty( (2*sim.nl,nm) )
        # The [0,0] component of "matrices" measures the size of a detector pixel in the 
        # simulated slit image space. i.e. slitmicrons/detpix.
        for i in range(sim.nl):
            self.square_offsets[:,i] = (pix_offset_ix - sim.nl/2.0) * sim.lenslet_width / self.matrices[i,self.x_map.shape[1]//2,0,0]
        self.sim_offsets = np.empty( (self.sim.im_slit_sz,nm) )
        #Creat an array of slit positions in microns. !!! Add an optional offset to this, i.e. a 1D offset !!!
        im_slit_pix_in_microns = (np.arange(self.sim.im_slit_sz) - self.sim.im_slit_sz/2.0) * self.sim.microns_pix
        for i in range(nm):
            self.sim_offsets[:,i] = im_slit_pix_in_microns / self.matrices[i,self.x_map.shape[1]//2,0,0]
        #To aid in 2D extraction, let's explicitly compute the y offsets corresponding to these x offsets...
        #The "matrices" map pixels back to slit co-ordinates. 
        self.slit_tilt = np.zeros( (nm,ny) )
        for i in range(nm):
            for j in range(ny):
                invmat = np.linalg.inv( self.matrices[i,j] )
                #What happens to the +x direction?
                x_dir_map = np.dot(invmat,[1,0])
                self.slit_tilt[i,j] = x_dir_map[1]/x_dir_map[0]
        
    def define_profile(self,fluxes):
        """ Manually define the slit profile as used in lenslet extraction. As this is
        a low-level function, all lenslets must be defined. e.g. by convention, for the
        star lenslets of the high resolution mode, lenslets 0,1 and 21 through 27 would 
        be zero. Also """
        
        if fluxes.shape[0] != self.sim.nl:
            print("Error: {0:s} resolution mode must have {1:d} lenslets".format(self.sim.mode,self.sim.nl))
        else:
            self.square_profile = np.empty( (fluxes.shape[0]*2, fluxes.shape[1]) )
            self.sim_profile = np.empty( (self.sim.im_slit_sz, fluxes.shape[1]) )
            for i in range(fluxes.shape[1]):
                self.square_profile[:,i] = np.array(fluxes[:,i]).repeat(2)
                im_slit=self.sim.make_lenslets(fluxes=fluxes[:,i])
                self.sim_profile[:,i] = np.sum(im_slit, axis=0)
        
    def one_d_extract(self, data=[], file='', lenslet_profile='sim', rnoise=3.0):
        """ Extract flux by integrating down columns (the "y" direction), using an
        optimal extraction method.
        
        Given that some of this code is in common with two_d_extract, the routines could
        easily be merged... however that would make one_d_extract less readable.
        
        Parameters
        ----------
        data: numpy array (optional) 
            Image data, transposed so that dispersion is in the "y" direction. Note that
            this is the transpose of a conventional echellogram. Either data or file
            must be given
            
        file: string (optional)
            A fits file with conventional row/column directions containing the data to be
            extracted.
        
        lenslet_profile: 'square' or 'sim'
            Shape of the profile of each fiber as used in the extraction. For a final
            implementation, 'measured' should be a possibility. 'square' assigns each
            pixel uniquely to a single lenslet. For testing only
        
        badpix: (float array, float array)
            Output of e.g. np.where giving the bad pixel coordinates.
        
        rnoise: float
            The assumed readout noise.
        
        WARNING: Binning not implemented yet"""
        
        if len(data)==0:
            if len(file)==0:
                print("ERROR: Must input data or file")
            else:
                if self.transpose_data:
                    #Transpose the data from the start.
                    data = pyfits.getdata(file).T
                else:
                    data = pyfits.getdata(file)
        
        ny = self.x_map.shape[1]
        nm = self.x_map.shape[0]
        nx = self.sim.szx
        
        #Number of "objects"
        no = self.square_profile.shape[1]
        extracted_flux = np.zeros( (nm,ny,no) )
        extracted_var = np.zeros( (nm,ny,no) )
        
        #Assuming that the data are in photo-electrons, construct a simple model for the
        #pixel inverse variance.
        pixel_inv_var = 1.0/(np.maximum(data,0)/self.sim.gain + rnoise**2)
        pixel_inv_var[self.badpixmask]=0.0
                
        #Loop through all orders then through all y pixels.
        for i in range(nm):
            print("Extracting order: {0:d}".format(i))
            #Based on the profile we're using, create the local offsets and profile vectors
            if lenslet_profile == 'square':
                offsets = self.square_offsets[:,i]
                profile = self.square_profile
            elif lenslet_profile == 'sim':
                offsets = self.sim_offsets[:,i]
                profile = self.sim_profile
            nx_cutout = 2*int( (np.max(offsets) - np.min(offsets))/2 ) + 2
            phi = np.empty( (nx_cutout,no) )
            for j in range(ny):
                #Check for NaNs
                if self.x_map[i,j] != self.x_map[i,j]:
                    extracted_var[i,j,:] = np.nan
                    continue
                #Create our column cutout for the data and the PSF. !!! Is "round" correct on the next line??? 
                x_ix = int(np.round(self.x_map[i,j])) - nx_cutout//2 + np.arange(nx_cutout,dtype=int) + nx//2
                for k in range(no):
                    phi[:,k] = np.interp(x_ix - self.x_map[i,j] - nx//2, offsets, profile[:,k])
                    phi[:,k] /= np.sum(phi[:,k])
                #Deal with edge effects...
                ww = np.where( (x_ix >= nx) | (x_ix < 0) )[0]
                x_ix[ww]=0
                phi[ww,:]=0.0
                
                #Stop here. 
#                if i==10:
#                    pdb.set_trace()
            
                #Cut out our data and inverse variance.
                col_data = data[j,x_ix]
                col_inv_var = pixel_inv_var[j,x_ix]
                #Fill in the "c" matrix and "b" vector from Sharp and Birchall equation 9
                #Simplify things by writing the sum in the computation of "b" as a matrix
                #multiplication. We can do this because we're content to invert the 
                #(small) matrix "c" here. Equation 17 from Sharp and Birchall 
                #doesn't make a lot of sense... so lets just calculate the variance in the
                #simple explicit way.
                col_inv_var_mat = np.reshape(col_inv_var.repeat(no), (nx_cutout,no) )
                b_mat = phi * col_inv_var_mat
                c_mat = np.dot(phi.T,phi*col_inv_var_mat)
                pixel_weights = np.dot(b_mat,np.linalg.inv(c_mat))
                extracted_flux[i,j,:] = np.dot(col_data,pixel_weights)
                extracted_var[i,j,:] = np.dot(1.0/np.maximum(col_inv_var,1e-12),pixel_weights**2)
                #if ((i % 5)==1) & (j==ny//2):
                #if (i%5==1) & (j==ny//2):
                #if (j==ny//2):
                #    pdb.set_trace()
                    
        return extracted_flux, extracted_var
        
    def two_d_extract(self, file='', data=[], lenslet_profile='sim', rnoise=3.0, deconvolve=True):
        """ Extract using 2D information. The lenslet model used is a collapsed profile, 
        in 1D but where we take into account the slit shear/rotation by interpolating this
        1D slit profile to the nearest two pixels along each row (y-axis in code).
        
        One key difference to Sharp and Birchall is that c_kj (between equations 8 and 9)
        is the correct normalisation for a (fictitious) 1-pixel wide PSF centered exactly
        on a pixel, but not for a continuum. We normalise correctly for a continuum by
        having one of the \phi functions being one-pixel wide along the slit, and the 
        other being unbounded in the dispersion direction.
        
        Note that the input data has to be the transpose of a conventional echellogram
       
        TODO:
        1) Neaten the approximate matrix inverse square root
        
        Parameters
        ----------
        data: numpy array (optional) 
            Image data, transposed so that dispersion is in the "y" direction. Note that
            this is the transpose of a conventional echellogram. Either data or file
            must be given
            
        file: string (optional)
            A fits file with conventional row/column directions containing the data to be
            extracted.
        
        lenslet_profile: 'square' or 'sim'
            Shape of the profile of each fiber as used in the extraction. For a final
            implementation, 'measured' should be a possibility. 'square' assigns each
            pixel uniquely to a single lenslet. For testing only
        
        rnoise: float
            The assumed readout noise.
            
        deconvolve: bool
            Do we deconvolve so that neighboring extracted spectral points 
            are statistically independent? This is an approximate deconvolution (a linear 
            function of 5 neighboring pixels) so is reasonably robust. """
            
        if len(data)==0:
            if len(file)==0:
                print("ERROR: Must input data or file")
            else:
                #Transpose the data from the start.
                data = pyfits.getdata(file).T

        ny = self.x_map.shape[1]
        nm = self.x_map.shape[0]
        nx = self.sim.szx
        
        #Number of "objects"
        no = self.square_profile.shape[1]
        extracted_flux = np.zeros( (nm,ny,no) )
        extracted_var = np.zeros( (nm,ny,no) )
        extracted_covar = np.zeros( (nm,ny-1,no) )
        
        #Assuming that the data are in photo-electrons, construct a simple model for the
        #pixel inverse variance.
        pixel_inv_var = 1.0/(np.maximum(data,0) + rnoise**2)
        pixel_inv_var[self.badpixmask]=0.0
                
        #Loop through all orders then through all y pixels.
        for i in range(nm):
            print("Extracting order index: {0:d}".format(i))
            #Based on the profile we're using, create the local offsets and profile vectors
            if lenslet_profile == 'sim':
                offsets = self.sim_offsets[:,i]
                profile = self.sim_profile
            else:
                print("Only sim lenslet profile available for 2D extraction so far...")
                raise userwarning
            nx_cutout = 2*int( (np.max(offsets) - np.min(offsets))/2 ) + 2
            ny_cutout = 2*int(nx_cutout * np.nanmax(np.abs(self.slit_tilt)) / 2) + 3
            for j in range(ny):
                phi = np.zeros( (ny_cutout,nx_cutout,no) )
                phi1d = np.zeros( (ny_cutout,nx_cutout,no) )
                #Check for NaNs
                if self.x_map[i,j] != self.x_map[i,j]:
                    extracted_var[i,j,:] = np.nan
                    continue
                #Create our column cutout for the data and the PSF
                x_ix = int(self.x_map[i,j]) - nx_cutout//2 + np.arange(nx_cutout,dtype=int) + nx//2
                y_ix = j + np.arange(ny_cutout, dtype=int) - ny_cutout//2
                for k in range(no):
                    x_prof = np.interp(x_ix - self.x_map[i,j] - nx//2, offsets, profile[:,k])
                    y_pix = (x_ix - self.x_map[i,j] - nx//2) * self.slit_tilt[i,j] + ny_cutout//2
                    frac_y_pix = y_pix - y_pix.astype(int)
                    subx_ix = np.arange(nx_cutout,dtype=int)
                    phi[y_pix.astype(int),subx_ix,k] = (1-frac_y_pix)*x_prof
                    phi[y_pix.astype(int)+1,subx_ix,k] = frac_y_pix*x_prof
                    phi[:,:,k] /= np.sum(phi[:,:,k])  
                    x_prof /= np.sum(x_prof)               
                    phi1d[:,:,k] = np.tile(x_prof,ny_cutout).reshape( (ny_cutout, nx_cutout) )
                #Deal with edge effects...
                ww = np.where( (x_ix >= nx) | (x_ix < 0) )[0]
                x_ix[ww]=0
                phi[:,ww,:]=0.0
                phi1d[:,ww,:]=0.0
                ww = np.where( (y_ix >= ny) | (y_ix < 0) )[0]
                y_ix[ww]=0
                phi[ww,:,:]=0.0
                xy = np.meshgrid(y_ix, x_ix, indexing='ij') 
                #Cut out our data and inverse variance.
                col_data = data[xy].flatten()
                col_inv_var = pixel_inv_var[xy].flatten()
                #Fill in the "c" matrix and "b" vector from Sharp and Birchall equation 9
                #Simplify things by writing the sum in the computation of "b" as a matrix
                #multiplication. We can do this because we're content to invert the 
                #(small) matrix "c" here. Equation 17 from Sharp and Birchall 
                #doesn't make a lot of sense... so lets just calculate the variance in the
                #simple explicit way.
                col_inv_var_mat = np.reshape(col_inv_var.repeat(no), (ny_cutout*nx_cutout,no) )
                phi = phi.reshape( (ny_cutout*nx_cutout,no) )
                phi1d = phi1d.reshape( (ny_cutout*nx_cutout,no) )
                b_mat = phi * col_inv_var_mat
                c_mat = np.dot(phi.T,phi1d*col_inv_var_mat)
                pixel_weights = np.dot(b_mat,np.linalg.inv(c_mat))
#                if (j==1000):
#                        pdb.set_trace()
                extracted_flux[i,j,:] = np.dot(col_data,pixel_weights)
                extracted_var[i,j,:] = np.dot(1.0/np.maximum(col_inv_var,1e-12),pixel_weights**2)
                if (j > 0):
                    extracted_covar[i,j-1,:] = np.dot(1.0/np.maximum(col_inv_var,1e-12),pixel_weights* \
                        np.roll(last_pixel_weights,-nx_cutout, axis=0))
                last_pixel_weights = pixel_weights.copy()
#                if (j > 591):
#                    pdb.set_trace()
        if (deconvolve):
            #Create the diagonals of the matrix Q gradually, using the Taylor approximation for
            #the matrix inverse.
            #(Bolton and Schlegel 2009, equation 10)
            #D = diag(C)
            #A = D^{-1/2} (C-D) D^{-1/2}, so C = D^{1/2}(I + A)D^{1/2}
            #Then if Q = (I - 1/2 A + 3/8 A^2) D^{-1/2}
            #... then C^{-1} = QQ, approximately.
            #Note that all of this effort doesn't really seem to achieve much at all in practice...
            #an extremely marginal improvement in resolution... but at least formal pixel-to-pixel
            #data independence is returned.
            extracted_sig = np.sqrt(extracted_var)
            a_diag_p1 = extracted_covar/extracted_sig[:,:-1,:]/extracted_sig[:,1:,:]
#            a_diag_m1 = extracted_covar/extracted_var[:,1:,:]
            Q_diag = np.ones( (nm,ny,no) )
            Q_diag[:,:-1,:] += 3/8.0*a_diag_p1**2
            Q_diag[:,1:,:]  += 3/8.0*a_diag_p1**2
#            Q_diag[:,:-1,:] += 3/8.0*a_diag_p1*a_diag_m1
#            Q_diag[:,1:,:]  += 3/8.0*a_diag_p1*a_diag_m1
            Q_diag /= extracted_sig
            extracted_sqrtsig = np.sqrt(extracted_sig)
            Q_diag_p2 = 3/8.0*a_diag_p1[:,:-1,:]*a_diag_p1[:,1:,:]/extracted_sqrtsig[:,2:,:]/extracted_sqrtsig[:,:-2,:]
#            Q_diag_m2 = 3/8.0*a_diag_m1[:,:-1,:]*a_diag_m1[:,1:,:]/extracted_sig[:,:-2,:]
#            Q_diag_m1 = -0.5*a_diag_m1/extracted_sig[:,:-1,:]
            Q_diag_p1 = -0.5*a_diag_p1/extracted_sqrtsig[:,1:,:]/extracted_sqrtsig[:,:-1,:]
    #The approximation doesn't seem to be quite right, with the ~3% uncertainty on the diagonal of cinv, when there should
    #only be a ~1% uncertainty (obtained by going to the next term in the Taylor expansion). But pretty close...
    #Q = np.diag(Q_diag[0,:,0]) + np.diag(Q_diag_m1[0,:,0],k=-1) + np.diag(Q_diag_p1[0,:,0],k=+1) + np.diag(Q_diag_p2[0,:,0],k=+2) + np.diag(Q_diag_m2[0,:,0],k=-2)
    #cinv_approx = np.dot(Q,Q)
    #cinv = np.diag(extracted_var[0,:,0]) + np.diag(extracted_covar[0,:,0],k=1) + np.diag(extracted_covar[0,:,0],k=-1)
    #cinv = np.linalg.inv(cinv)
            #Now we have a sparse matrix with 5 terms. We need to sum down the rows, ignoring the 
            #edge pixels
#            s_vect = Q_diag[:,2:-2,:] + Q_diag_p1[:,1:-2,:] + Q_diag_m1[:,2:-1,:] + Q_diag_p2[:,:-2,:] + Q_diag_m2[:,2:,:]
            s_vect = Q_diag.copy()
            s_vect[:,:-1,:] += Q_diag_p1
            s_vect[:,:-2,:] += Q_diag_p2
            s_vect[:,1:,:] += Q_diag_p1
            s_vect[:,2:,:] += Q_diag_p2
            new_var = 1.0/s_vect**2
            new_flux = extracted_flux*Q_diag/s_vect
            new_flux[:,:-1,:] += extracted_flux[:,1:,:]*Q_diag_p1/s_vect[:,1:,:]
            new_flux[:,:-2,:] += extracted_flux[:,2:,:]*Q_diag_p2/s_vect[:,2:,:]
            new_flux[:,1:,:] += extracted_flux[:,:-1,:]*Q_diag_p1/s_vect[:,:-1,:]
            new_flux[:,2:,:] += extracted_flux[:,:-2,:]*Q_diag_p2/s_vect[:,:-2,:]
            
            #Fill in the Variance and Flux arrays with NaNs, so that the (not computed) edges 
            #are undefined.
 #           new_flux = np.empty_like(extracted_flux)
 #           new_var = np.empty_like(extracted_var)
 #           new_flux[:,:,:]=np.nan
 #           new_var[:,:,:]=np.nan
            #Now fill in the arrays.
 #           new_var[:,2:-2,:] = 1.0/s_vect**2
 #           new_flux[:,2:-2,:] =  extracted_flux[:,2:-2,:]*Q_diag[:,2:-2,:]/s_vect 
            #
 #           new_flux[:,2:-2,:] += extracted_flux[:,1:-3,:]*Q_diag_p1[:,1:-2,:]/s_vect
 #           new_flux[:,2:-2,:] += extracted_flux[:,3:-1,:]*Q_diag_p1[:,2:-1,:]/s_vect
 #           new_flux[:,2:-2,:] += extracted_flux[:,:-4,:] *Q_diag_p2[:,:-2,:]/s_vect
 #           new_flux[:,2:-2,:] += extracted_flux[:,4:,:]  *Q_diag_p2[:,2:,:]/s_vect
            
            return new_flux, new_var
        else:
            return extracted_flux, extracted_var
        
        
    def find_lines(self,data,arcfile='lines.txt',outfile='arclines.txt', hw=10,flat_data=[]):
        """Find lines near the locations of input arc lines.
        
        Parameters
        ----------
        data: numpy array
            data array
            
        arcfile: string
            file containing lines """

        #First, extract the data
        flux,var = self.one_d_extract(data=data, rnoise=self.sim.rnoise)
        #Read in the lines
        lines = np.loadtxt(arcfile)
        #Only use the first lenslet.
        flux = flux[:,:,0]
        ny = self.x_map.shape[1]
        nm = self.x_map.shape[0]
        nx = self.sim.szx
        lines_out=[]
        if len(flat_data)>0:
            data_to_show = data - 0.05*flat_data
        else:
            data_to_show = data.copy()
        plt.clf()
        plt.imshow( np.arcsinh( (data_to_show-np.median(data_to_show))/1e2) , interpolation='nearest', aspect='auto', cmap=cm.gray)
        for m_ix in range(nm):
            w_ix = np.interp(lines,self.w_map[m_ix,:],np.arange(ny))
            ww = np.where( (w_ix >= hw) & (w_ix < ny-hw) )[0]
            w_ix = w_ix[ww]
            arclines_to_fit = lines[ww]
            for i,ix in enumerate(w_ix):
                x = np.arange(ix-hw,ix+hw,dtype=np.int)
                y = flux[m_ix,x]
                y -= np.min(y) #Rough...
                if np.max(y)< 25*self.sim.rnoise:
                    continue
                g_init = models.Gaussian1D(amplitude=np.max(y), mean=x[np.argmax(y)], stddev=1.5)
                fit_g = fitting.LevMarLSQFitter()
                g = fit_g(g_init, x, y)
                #Wave, ypos, xpos, m, amplitude, fwhm
                xpos = nx//2+np.interp(g.mean.value,np.arange(ny),self.x_map[m_ix])
                ypos = g.mean.value
                plt.plot(xpos,ix,'bx')
                plt.plot(xpos,ypos,'rx') #!!! Maybe around the other way?
                plt.text(xpos+10,ypos,str(arclines_to_fit[i]),color='green',fontsize=10)
                lines_out.append( [arclines_to_fit[i],ypos,xpos,m_ix+self.sim.m_min,g.amplitude.value, g.stddev.value*2.3548] )
        plt.axis([0,nx,ny,0])
        lines_out = np.array(lines_out)
        np.savetxt(outfile,lines_out,fmt='%9.4f %7.2f %7.2f %2d %7.1e %4.1f')
        
