"""This is an image calibration module using only routines available in astropy, ideally. 

Check each individual function for instructions"""

#Things needed: 
#image combination function (biass, darks, flats, (arcs?) to generate masters with sigma clipping, simple stat analysis to determine what is what
#image subtraction function
#image list population (perhaps pywifes style)
#Wrapper function for image calibration
#pyDrizzle can be used for median combine
#pyfits will be used for image subtraction
#pickle may be used for image dictionary format, but perhaps not a bad idea to contain a text file for visual inspection as well

from astropysics import ccd
try: import pyfits
except: import astropy.io.fits as pyfits
import commands, pickle
import numpy as np



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

    def updateHeaderFormat(self,headerKey,headerValue,frametype, reset=False):
        """ This function is used to define what header keyword/value pair to use to identify the frame types. The idea behind this is that the pipeline should have knowledge of how each frame type is distinguished in the header keywords. This function will allow the users to provide header keyword:value pairs for the frame identification.

        e.g. If the bias images are flagged as IMAGETYP='Bias', an 'IMAGETYP','Bias' pair would be provided to this function to add this pair to the existing list

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
        

    def makeImageList(self,path_to_files='./'):
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
        
    def imageCombine(self, imagelist, output, method='median', sigmaclip=None):
        """ Function to do image combination, useful for combining e.g. darks and baises. STILL NOT FULLY TESTED.

        Parameters
        ----------

        imageList: string
             Either a frame type from self.frametypes class attribute (e.g. 'bias') or a file name of a file containng a list of images. 
        output: string
            Output file name for the combined image
        method: string (optional)
            Combination method. Either 'mean', 'median' (default) or 'sum'
        sigmaclip: (optional)
            How many sigma away from average before point is removed from combination. Defaults to none
        
        Returns
        -------
        s: string indicating success or failure
        """
        #Import imageCombiner class from astropysics.ccd
        ic=ccd.ImageCombiner()
        if not method in ['mean','median','sum']:
            return 'Invalid combination method'
        ic.method=method
        if sigmaclip: 
            try: float(sigmaclip)
            except Exception: return 'Invalid sigma clip value'
            ic.sigclip=sigmaclip
        #check if imagelist is part of the frametypes we accept
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
        try:
            comb=ic.combineImages(imagecube)
        except Exception:
            return 'Unable to combine images. Possibly not enough memory or images are not the same shape.'
        h.add_comment('Result of combining '+imagelist+' images')
        pyfits.writeto(output,comb)#,header=h)
        return 'Successfully combined images.'


        
    
