"""This is an image calibration module using only routines available in astropy, 
ideally. 

Check each individual function for instructions"""

#Things needed: 
#1) image combination function (biass, darks, flats, (arcs?) to generate masters 
#   with sigma clipping, simple stat analysis to determine what is what
#2) image subtraction function
#3) image list population (perhaps pywifes style)
#4) Wrapper function for image calibration
#5) pyDrizzle can be used for median combine
#6) pyfits will be used for image subtraction
#7) pickle may be used for image dictionary format, but perhaps not a bad idea 
#   to contain a text file for visual inspection as well

try: import pyfits
except: import astropy.io.fits as pyfits
import commands, pickle
import numpy as np
import os
from distutils.version import StrictVersion

class Calibration():
    """ A class for performing basic frame calibrations. 
    Based almost entirely on the astropysics module for the image combination. 
    It's a series of functions to do things.

    Should have modules for image combination, image subtraction, 
    flatfielding, identification of images based on header 
    keywords and statistical analysis of images.
    """
    bias_headers=[('IMAGETYP','zero'),('IMAGETYP','bias'),('OBJECT','bias')]
    dark_headers=[('IMAGETYP','dark'),('OBJECT','dark')]
    lampflat_headers=[('IMAGETYP','flat'),('OBJECT','flat')]
    skyflat_headers=[('IMAGETYP','skyflat')]
    arc_headers=[('IMAGETYP','arc')]
    std_headers=[('NOTES','RV Stardard')]
    sci_headers=[('IMAGETYP','light')]
    frametypes=['bias', 'dark', 'lampflat','skyflat','arc','std','sci']

    def update_header_format(self,headerKey,headerValue,frametype, reset=False):
        """ This function is used to define what header keyword/value pair to 
        use to identify the frame types. The idea behind this is that the 
        pipeline should have knowledge of how each frame type is distinguished 
        in the header keywords. This function will allow the users to provide 
        header keyword:value pairs for the frame identification.

        e.g. If the bias images are flagged as IMAGETYP='Bias', an 'IMAGETYP',
        'Bias' pair would be provided to this function to add this pair to the 
        existing list

        Parameters
        ----------
        headerKey: string
            Header keyword containing the distinguishing factor between frames.
        headerValue: Undefined type
            Header Value for the frame type defined by frametype.
        frametype: string
            Type of frame. Can be 'bias', 'dark', 'lampflat','skyflat','sci' or 'arc'
        reset: boolean (optional)
            If True, all other previous options are erased.
        
        Returns
        -------
        A success statement or failure statement. 
        """
        if not frametype in self.frametypes:
            return 'Invalid frame type selection.'
        try:
            dummy=str(headerKey)
        except Exception: return 'Invalid header keyword option. Must be a string'
        if reset:
            try: exec('self.'+frametype+'_headers=[(\"'+headerKey+'\",\"'+headerValue+'\")]')
            except Exception: return 'Unable to reset headers'
        else:
            try: eval('self.'+frametype+'_headers').append((headerKey,headerValue))
            except Exception: return 'Unable to add the ',headerKey,headerValue,' pair to the ',frametype,' header keyword list'
        return 'Successfully added the ',headerKey,headerValue,' pair to the ',frametype,' header keyword list'
        

    def make_image_list(self,path_to_files='./'):
        """ Function to make an easily importable file 
        containing each file type associated with file names.

        Parameters
        ----------
        path_to_files: string (optional)
            Path to the images being surveyed for this reduction. 
            Default is current directory.

        Returns
        -------
        s: string
            A visual representation of the contents of the file created. 
            The actual file will be saved in the current directory.

        """
        bias_list=[]
        dark_list=[]
        skyflat_list=[]
        lampflat_list=[]
        arc_list=[]
        sci_list=[]
        std_list=[]
        #Assume only fits files in directory are of interest
        ll=commands.getoutput('ls '+path_to_files+'*.fit*').split('\n')
        if len(ll)==0:
            return 'Directory specified does not contain any fits files.'
        #Now run through all the files and try to find out what each are.
        for fi in ll:
            current_file_header=pyfits.getheader(fi)
            for ft in self.frametypes:
                success=False
                for headerinfo in eval('self.'+ft+'_headers'):
                    try: 
                        headerinfo[1].upper()
                        if current_file_header[headerinfo[0]].upper()==headerinfo[1].upper():
                            success=True
                            eval(ft+'_list').append(fi)
                            break
                    except:
                        if current_file_header[headerinfo[0]].upper()==headerinfo[1]:
                            success=True
                            eval(ft+'_list').append(fi)
                            break
                if success: break

        for i in self.frametypes:
            print 'List of ',i,' frames: ',eval(i+'_list')
        print 'Check that this is correct and only then continue'
        print 'Saving List into frame type file'
        night_data=night_data = {
            'bias' : bias_list,
            'skyflat' : skyflat_list,
            'dark' : dark_list,
            'lampflat' : lampflat_list,
            'arc'  : arc_list,
            'sci'  : sci_list,
            'std'  : std_list}
        f1 = open('frame_types.pkl', 'w')
        pickle.dump(night_data, f1)
        f1.close()
        
    def image_combine(self, output, imagelist=None, method='median', sigmaclip=None,overwrite=False):
        """ Function to do image combination, useful for combining. e.g. darks 
        and baises. If you want to median combine with sigma clipping, you need 
        numpy 1.9 or above.

        Parameters
        ----------

        imageList: string
            Either a frame type from self.frametypes class attribute 
            (e.g. 'bias') or a file name of a file containing a list of images 
            or simply a python list with file names. 
        output: string
            Output file name for the combined image
        method: string (optional)
            Combination method. Either 'mean', 'median' (default) or 'sum'
        sigmaclip: (optional)
            How many sigma away from average before point is removed from 
            combination. Defaults to none
        
        Returns
        -------
        s: string indicating success or failure
        """
        #Check if output file exists. If so and if overwrite is False, return a warning and don't combine.
        if os.path.exists(output) and not overwrite:
            return 'Will not try to combine since output file exists and overwrite option is set to False.'
        
        if not method in ['mean','median','sum']:
            return 'Invalid combination method'
        if sigmaclip: 
            try:
                float(sigmaclip)
            except Exception: return 'Invalid sigma clip value'
            if method=='median' and (StrictVersion(np.__version__) < StrictVersion('1.9.0')):
                return 'You need numpy version 1.9.0 or above for median combining with sigma clipping'
        #check if imagelist is part of the frametypes we accept
        if type(imagelist)==str:
            if imagelist in self.frametypes:
                print 'Assuming you have already ran makeImageList and grabbing file names from there'
                f=open('frame_types.pkl')
                night_data=pickle.load(f)
                f.close()
                images=night_data[imagelist]
            #Otherwise, look for an input file
            else:
                try: 
                    images=np.loadtxt(imagelist,dtype='str')
                except Exception:
                    return 'Failed to load image list from file '+imagelist+'. Make sure only file names exist and that they are on separate lines.'
        elif type(imagelist)==list:
            images=imagelist
        elif images is None:
             return 'Must input imagelist or images'
        #Start loading images onto list
        shape=np.shape(pyfits.getdata(images[0]))
        h=pyfits.getheader(images[0])
        imagecube=[]
        comb_index=1
        for i in images:
            d=pyfits.getdata(i)
            if np.shape(d)==shape:
                imagecube.append(d)
            else: return 'Image combination failed! Image '+i+' has a different shape to the others on the list.'
            h.append(('COMB'+str(comb_index),i,'Image used for combination'))
            comb_index+=1
        if method == 'median':
            op = np.nanmedian
        elif method == 'mean':
            op = np.nanmean
        elif method == 'sum':
            op = np.nansum
        if sigmaclip is not None:
            mean_ar=np.mean(imagecube,axis=0)
            std_ar=np.std(imagecube,axis=0)
            sds = imagecube-mean_ar
            sds/=std_ar
            imagecube = np.ma.masked_where(sds>sigmaclip,imagecube)
            imagecube.fill_value=np.nan
            imagecube=imagecube.data
            del sds
        try:
            comb = op(imagecube,axis=0)
        except TypeError:
            comb = op(imagecube)
        except Exception:
            return 'Unable to combine images. Possibly not enough memory or images are not the same shape.'
        if imagelist:
            h.add_comment('Result of combining a series of images')
        if os.path.exists(output) and overwrite:
            os.system('rm '+output)
        pyfits.writeto(output,comb,header=h)
        return 'Successfully combined images.'


    def image_subtraction(self,image,calframe,output='./',postfix=None,overwrite=False):
        """
        Function to do image subtraction, useful for bias and/or dark frame 
        calibration. UNTESTED SO FAR

        Parameters
        ----------

        image: string or list
            Original fits image to be calibrated or list of images to be 
            calibrated. If a string with '.fits' is provided, only a single 
            image is calibrated. If a string without the extension is provided, 
            pipeline looks for a file with a list of images in the path 
            provided. if a python list is provided, pipeline will run through 
            all and calibrate all.
        calframe: string
            Calibration fits frame to be subtracted.
        output: string
            If it is a directory, determined by the '/' character in the name, 
            all files will retain their original input name and be placed inside 
            that folder with (optionally) a postfix added to the file name. If 
            it is a single file name, this is the output file name. 
        postfix: string (optional)
            Postfix to add to the fits images post calibration. e.g., if 
            original image is test.fits, and postfix is '_biased', final image 
            is test_biased.fits.
        overwrite: Boolean (optional)
            If True, any image encountered that already exists at the output 
            saving stage will be overwritten.
        
        Returns
        -------
        s: string indicating success or failure
        """
        #First figure out what the image input is and create a list to calibrate
        if type(image)==list:
            images=image
        elif type(image)==str:
            if '.fit' in image:
                images=[image]
            else:
                try: images=np.loadtxt(image,dtype='str')
                except Exception: return 'Unable to load image list from file ',image
        else: return 'Unable to determine what the image(s) to calibrate are.'
        #Now that the images list is created load the calibration image
        try: calimage=pyfits.getdata(calframe)
        except Exception: return 'Failed to load calibration image'
        #Now determine if the outut is a single image or a folder.
        if '/' in output:
            try: 
                os.system('mkdir '+output)
                single_output=False
            except Exception: pass
        else: 
            single_output=True
        #Now do the calibration
        for im in images:
            try: 
                f=pyfits.open(im)
                h=f[0].header
                d=f[0].data
            except Exception: return 'Unable to open file ',im
            if single_output and len(images)<2:
                return 'Multiple images for calibration but single output specified. Ambiguous...'
            if np.shape(d)!=np.shape(calimage):
                return 'Image '+im+' can not be calibrated due to incompatible size with the calibration image.'
            outdata=d-calimage
            h.add_comment('Image calibrated using '+calframe)
            pf=''
            if postfix:
                pf=postfix
            if single_output:
                outname=output
            else: 
                outname=output+im.split('.fit')[0]+pf+'.fits'
            #Now that the output name is defined and the image is subtracted, 
            #along side a comment added on the header keyword, try to write new 
            #file onto system.
            try: 
                dummy=pyfits.writeto(outname,outdata,header=h)
            except Exception:
                return 'Unable to write image to disk'
        return 'Successfully calibrated image(s)'

    def make_badpix_mask(self,bias_list=[],dark_list=[],flat_list=[],outfile='badpix.fits',sigma=3.0,delete_temp=True):
        """
        Function useful to create a bad pixel mask. It uses a set of biases, 
        darks (ideally with different exposure times) and flats (also ideally 
        with different exposure times) and looks for bad/hot/non-linear pixels 
        that fall outside of a certain specified number of sigma from the average.

        Parameters
        ----------

        bias_list: python list (optional)
            List of bias files to use. 
        dark_list: python list (optional)
            List of dark frames.
        flat_list: python list (optional)
            List of flat field frames.
        output: string (optional)
            Output file name containing list of bad pixels. Defaults to 'badpix'.
        sigma: float (optional)
            Sigma used during the determination of which pixels to consider bad. Default 3.0.
        delete_temp: boolean (optional)
            Boolean indicating whether temporary files created are deleted.
        
        Returns
        -------
        s: string indicating success or failure
        """
        no_darks=False
        no_bias=False
        no_flats=False
        #Check if any lists are provided
        if bias_list==[] and dark_list==[] and flat_list==[]:
            return 'You must supply at least one set of images'
        #Try to create a temporary folder for combined images
        temp_dir='temp/'
        try:
            os.system('mkdir temp/;rm temp/*')
        except Exception: return 'Unable to create temporary folder for temp files'
        #If bias list was supplied, combined into master bias:
        if bias_list!=[]:
            if len(bias_list)>1:
                result=self.image_combine(output=temp_dir+'mbias.fits', imagelist=bias_list, method='mean', sigmaclip=None,overwrite=True)
                if 'Successfully' not in result:
                    return 'Unable to combine biases'
                mbias=temp_dir+'mbias.fits'
            else:
                mbias=bias_list[0]
            bias=pyfits.getdata(mbias)
        else:
            bias=0
            no_bias=True

        #Now look at darks. Combine sets of high and low exposures if enough available.
        if dark_list!=[]:
            if len(dark_list)>1:
                dark_list=np.array(dark_list)
                #Look at fits files for exposure times and work out min and max
                exposure_times=[]
                for i in dark_list:
                    try: exposure_times.append(pyfits.getheader(i)['EXPTIME'])
                    except Exception: exposure_times.append(pyfits.getheader(i)['EXPOSURE'])
                exposure_times=np.array(exposure_times)
                #Combine darks with lowest exposure time
                result=self.image_combine(output=temp_dir+'mdark_min.fits', imagelist=dark_list[exposure_times==np.min(exposure_times)].tolist(), method='mean', sigmaclip=None,overwrite=True)
                if 'Successfully' not in result:
                    return 'Unable to combine first set of darks'
                dark_min=pyfits.getdata(temp_dir+'mdark_min.fits')-bias

                if len(np.unique(exposure_times))>1:
                    #Combine darks with highest exposure time
                    result=self.image_combine(output=temp_dir+'mdark_max.fits', imagelist=dark_list[exposure_times==np.max(exposure_times)].tolist(), method='mean', sigmaclip=None,overwrite=True)
                    if 'Successfully' not in result:
                        return 'Unable to combine second set of darks'
                    dark_max=pyfits.getdata(temp_dir+'mdark_max.fits')-bias
                else:
                    dark_max=dark_min
            else:
                dark_min=pyfits.getdata(dark_list[0])-bias
                dark_max=dark_min
        else:
            no_darks=True
        

        #Now look at flats. Combine sets of high and low exposures if enough available.               
        if flat_list!=[]:
            if len(flat_list)>1:
                flat_list=np.array(flat_list)
                #Look at fits files for exposure times and work out min and max
                exposure_times=[]
                for i in flat_list:
                    try: exposure_times.append(pyfits.getheader(i)['EXPTIME'])
                    except Exception: exposure_times.append(pyfits.getheader(i)['EXPOSURE'])
                exposure_times=np.array(exposure_times)
              
                #Combine flats with lowest exposure time
                result=self.image_combine(output=temp_dir+'mflat_min.fits', imagelist=flat_list[exposure_times==np.min(exposure_times)].tolist(), method='mean', sigmaclip=None,overwrite=True)
                if 'Successfully' not in result:
                    return 'Unable to combine first set of flats'
                flat_min=pyfits.getdata(temp_dir+'mflat_min.fits')-bias

                if len(np.unique(exposure_times))>1:
                    #Combine flats with highest exposure time
                    result=self.image_combine(output=temp_dir+'mflat_max.fits', imagelist=flat_list[exposure_times==np.max(exposure_times)].tolist(), method='mean', sigmaclip=None,overwrite=True)
                    if 'Successfully' not in result:
                        return 'Unable to combine second set of flats'
                    flat_max=pyfits.getdata(temp_dir+'mflat_max.fits')-bias
                else:
                    flat_max=flat_min
            else:
                flat_min=pyfits.getdata(flat_list[0])-bias
                flat_max=flat_min
        else:
            no_flats=True

        #Now start figuring out what is bad
        badpix=np.zeros(1)
        #If we have any biases
        if np.size(bias)>1:
            badpix=np.zeros(np.shape(bias))
            mean=np.average(bias)
            std=np.std(bias)
            badpix[(bias>mean+std*sigma)|(bias<mean-std*sigma)]=1
        #if we have darks
        if not no_darks:
            if np.size(badpix)==1:
                badpix=np.zeros(np.shape(dark_max))
            mean=np.average(dark_max)
            std=np.std(dark_max)
            badpix[(dark_max>mean+std*sigma)|(dark_max<mean-std*sigma)]=1
            #divide darks and look for outliers
            dark_min[dark_min==0]=1
            darkdiv=np.float32(dark_max)/dark_min
            pyfits.writeto(temp_dir+'darkdiv.fits',darkdiv)
            mean=np.average(darkdiv)
            std=np.std(darkdiv)
            badpix[(darkdiv>mean+std*sigma)|(darkdiv<mean-std*sigma)]=1
        #if we have flats
        if not no_flats:
            if np.size(badpix)==1:
                badpix=np.zeros(np.shape(flat_max))
            #divide flats and look for outliers
            flat_min[flat_min==0]=1
            flatdiv=np.float32(flat_max)/flat_min
            pyfits.writeto(temp_dir+'flatdiv.fits',flatdiv)
            mean=np.average(flatdiv)
            std=np.std(flatdiv)
            badpix[(flatdiv>mean+std*sigma)|(flatdiv<mean-std*sigma)]=1
        try: 
            pyfits.writeto(outfile,badpix,clobber=True)
        except Exception:
            print 'Unable to save badpixel file'
            return 'Unable to save badpixel file'
        if delete_temp: os.system('rm -R temp/')
        return 'Successfully created a bad pixel mask'
        
        
