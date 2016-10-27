import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import pymfe,pdb
import astropy.io.fits as pyfits


data=pyfits.getdata('/home/jbento/flat_blue.fits')

ghost=pymfe.ghost.Arm('blue',mode='std')
wparams = np.loadtxt('./data/subaru/wavemod.txt')
xparams = np.loadtxt('./data/subaru/xmod.txt')

xparams= np.zeros_like(xparams)


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

q00i=2000.
q01i=2000.

y=np.arange(data.shape[1])
xparams[2,4]=q00i
xparams[2,3]=q01i
x_int,wave_int,blaze_int=ghost.spectral_format(wparams=wparams,xparams=xparams)
l, = plt.plot(x_int, y, lw=2, color='red')
#plt.axis([0, 1, -10, 

plt.imshow(ghost.slit_flat_convolve(flatfield=data))

axq00 = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
axq01 = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)


sq00 = Slider(axq00, 'q00', 0, 4000.0, valinit=q00i)
sq01 = Slider(axq01, 'q01', 0, 4000.0, valinit=q01i)


def update(val):
    q01 = s01.val
    q00 = sq00.val
    xparams[2,4]=q00
    xparams[2,3]=q01
    x,wave,blaze=ghost.spectral_format(wparams=wparams,xparams=xparams)
    l.set_ydata(q01*np.sin(2*np.pi*q00*t))
    fig.canvas.draw_idle()
sq00.on_changed(update)
sq01.on_changed(update)

#resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
#button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sq00.reset()
    sq01.reset()
button.on_clicked(reset)


plt.show()
