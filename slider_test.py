import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import pymfe,pdb
import astropy.io.fits as pyfits

datafile='/home/jbento/flat_blue.fits'
data=pyfits.getdata(datafile)
nx=data.shape[0]
ghost=pymfe.ghost.Arm('blue',mode='std')
wparams = np.loadtxt('./data/ghost/wavemod.txt')
xparams = np.loadtxt('./data/ghost/xmod.txt')

#xparams= np.zeros_like(xparams)


fig, ax = plt.subplots()
#plt.subplots_adjust(left=0, bottom=0.40)


x_int,wave_int,blaze_int=ghost.spectral_format(wparams=wparams,xparams=xparams)
y=np.meshgrid(np.arange(data.shape[1]),np.arange(x_int.shape[0]))[0]
#pdb.set_trace()
l, = plt.plot(y.flatten()[::10], x_int.flatten()[::10]+nx//2, color='green',\
    linestyle='None',marker='.' )

flat_conv=ghost.slit_flat_convolve(flatfield=datafile)
ax.imshow( (flat_conv-np.median(flat_conv))/1e2)

#plt.imshow(data)

axcolor = 'lightgoldenrodyellow'

#Create a second window for cliders.
slide_fig=plt.figure()

#This function is executed on each slider change. spectral_format is updated.
def update(val):
    for i in range(npolys):
        for j in range(polyorder):
            xparams[i,j]=sliders[i][j].val
    x,wave,blaze=ghost.spectral_format(wparams=wparams,xparams=xparams)
    l.set_ydata(x.flatten()[::10]+nx//2)
    fig.canvas.draw_idle()

polyorder=xparams.shape[1] #5
npolys=xparams.shape[0] #3
#Now we start putting sliders in depending on number of parameters
height=1./(npolys*2)
width=1./(polyorder*2)
#Use this to adjust in a percentage how much to let each parameter vary 
percentage_variation=0.1
frac_xparams=np.absolute(xparams*percentage_variation)
axq= [[0 for x in range(polyorder)] for y in range(npolys)] 
sliders=[[0 for x in range(polyorder)] for y in range(npolys)] 
#Now put all the sliders in the new figure based on position in the array
for i in range(npolys):
    for j in range(polyorder):
        left=j*width*2
        bottom=1-(i+1)*height*2+height
        axq[i][j] = plt.axes([left, bottom, width,height],axisbg=axcolor)
        if xparams[i,j]==0:
            sliders[i][j] = Slider(axq[i][j], 'test'+str(i)+str(j), 0, 0.1,\
                valinit=xparams[i,j])
        else:
            sliders[i][j] = Slider(axq[i][j], 'test'+str(i)+str(j),\
                xparams[i,j]-frac_xparams[i,j], \
                xparams[i,j]+frac_xparams[i,j],valinit=xparams[i,j])
        plt.legend(loc=3)
        sliders[i][j].on_changed(update)

#Have a button to output the current varied model to the correct file
submitax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(submitax, 'Submit', color=axcolor, hovercolor='0.975')

def submit(event):
    np.savetxt('data/ghost/xmod.txt',xparams,fmt='%.4e')
    print 'Data updated on xmod.txt'
button.on_clicked(submit)

"""THINGS TO ADD: 
-All outputs of each procedure must be fits files, including the polynomial fit parameters. 
-Reset reference to middle of the frame instead of bottom
-A fit button? 
"""

plt.show()
