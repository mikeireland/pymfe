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


class Calibration():
    """ A class for performing basic frame calibrations. 
    Based almost entirely on the astropysics module for the image combination. 
    It's a series of functions to do things.

    Should have modules for image combination, image subtraction, 
    flatfielding, identification of images based on header 
    keywords and statistical analysis of images.

    """
    bias_headers=[('IMAGETYP','zero'),('IMAGETYP','bias'),('OBJECT','bias')]
    dark_headers=[('IMAGETYP','dark')]
    lampflat_headers=[('IMAGETYP','flat')]
    skyflat_headers=[('IMAGETYP','skyflat')]
    arc_headers=[('IMAGETYP','arc')]
    sci_headers=[('IMAGETYP','light')]
    frametypes=['bias', 'dark', 'lampflat','skyflat','arc','sci']

    def updateHeaderFormat(self,headerKey,headerValue,frametype='bias', reset=False):
        """ This function is used to define what header keyword/value pair to use to identify the frame types. The idea behind this is that the pipeline should have knowledge of how each frame type is distinguished in the header keywords. This function will allow the users to provide header keyword:value pairs for the frame identification.

        e.g. If the bias images are flagged as IMAGETYP='Bias', an 'IMAGETYP','Bias' pair would be provided to this function to add this pair to the existing list

        Parameters
        ----------
        headerKey: string
            Header keyword containing the distinguishing factor between frames.
        headerValue: Undefined type
            Header Value for the frame type defined by frametype.
        frametype: string (optional)
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
            try: eval('self.'+frametype+'_headers')=[(headerKey,headerValue)]
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
                    if current_file_header[headerinfo[0]]==headerinfo[1]:
                        success=True
                        eval(ft+'_list').append(fi)
                        break
                if success: break

        print bias_list, dark_list, arc_list, sci_list
            
            
        
    
