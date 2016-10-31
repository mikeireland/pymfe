import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import pymfe,pdb
import astropy.io.fits as pyfits

datafile='/home/jbento/flat_blue.fits'
data=pyfits.getdata(datafile)

ghost=pymfe.ghost.Arm('blue',mode='std')
wparams = np.loadtxt('./data/subaru/wavemod.txt')
xparams = np.loadtxt('./data/subaru/xmod.txt')

xparams= np.zeros_like(xparams)


fig, ax = plt.subplots()
plt.subplots_adjust(left=0, bottom=0.40)

q00i=7548.
q01i=14663.46
q10i=.02
q11i=0.
q20i=0.0001

xparams[2,4]=q00i
xparams[2,3]=q01i
xparams[1,4]=q10i
xparams[1,3]=q11i
xparams[0,4]=q20i
x_int,wave_int,blaze_int=ghost.spectral_format(wparams=wparams,xparams=xparams)
y=np.meshgrid(np.arange(data.shape[1]),np.arange(x_int.shape[0]))[0]
#pdb.set_trace()
l, = plt.plot(y.flatten()[::10], x_int.flatten()[::10], color='green',\
    linestyle='None',marker='.' )


plt.imshow(ghost.slit_flat_convolve(flatfield=datafile))
#plt.imshow(data)
axcolor = 'lightgoldenrodyellow'
axq00 = plt.axes([0.2, 0.1, 0.7, 0.03], axisbg=axcolor)
axq01 = plt.axes([0.2, 0.15, 0.7, 0.03], axisbg=axcolor)
axq10 = plt.axes([0.2, 0.20, 0.7, 0.03], axisbg=axcolor)
axq11 = plt.axes([0.2, 0.25, 0.7, 0.03], axisbg=axcolor)
axq20 = plt.axes([0.2, 0.3, 0.7, 0.03], axisbg=axcolor)


sq00 = Slider(axq00, 'Reference position', 0, 15000.0, valinit=q00i)
sq01 = Slider(axq01, 'Order Spacing', 0, 100000.0, valinit=q01i)
sq10 = Slider(axq10, 'Order Slope', 0, 1.0, valinit=q10i)
sq11 = Slider(axq11, 'Order slope#2', 0, 1.0, valinit=q11i)
sq20 = Slider(axq20, 'Distortion', 0, 0.001, valinit=q20i)


def update(val):
    q01 = sq01.val
    q00 = sq00.val
    q10 = sq10.val
    q11 = sq11.val
    q20 = sq20.val
    
    xparams[2,4]=q00
    xparams[2,3]=q01
    xparams[1,4]=q10
    xparams[1,3]=q11
    xparams[0,4]=q20
    x,wave,blaze=ghost.spectral_format(wparams=wparams,xparams=xparams)
    l.set_ydata(x.flatten()[::10])
    fig.canvas.draw_idle()
sq00.on_changed(update)
sq01.on_changed(update)
sq10.on_changed(update)
sq11.on_changed(update)
sq20.on_changed(update)

#resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
#button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


# def reset(event):
#     sq00.reset()
#     sq01.reset()
# button.on_clicked(reset)

"""THINGS TO ADD: 
-All outputs of each procedure must be fits files, including the polynomial fit parameters. 
-Reset button
-Submit Button (and file export)
-Reset reference to middle of the frame instead of bottom
-A fit button? 
-Potentially Indexing variables to make sure variable numbers of sliders can be placed.
"""

plt.show()
