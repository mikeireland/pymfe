# This script combines RVs for Th/Ar and star observations
# It weights the nearest nref Th/Ar velocities by time**weight_exponent,
# finds the weighted average and subtracts this weighted velocity from
# each target velocity.

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

star_prefix = "sun_1072"
thar_prefix = "thar-_17"

weight_exponent = -0.5
nref = 6
m_min=2
m_max=28
#m_min=3
#m_max=19

#Load
dd = np.loadtxt(star_prefix + "_bcor_rv.csv", delimiter=",")
rvs = dd[:,1:14]
mjds = dd[:,0]
dd = np.loadtxt(star_prefix + "_rv_sig.csv", delimiter=",")
rv_sigs = dd[:,1:14]
dd = np.loadtxt(thar_prefix + "_rvs.csv", delimiter=",")
thar_rvs = dd[:,1:14]
thar_mjds = dd[:,0]
dd = np.loadtxt(thar_prefix + "_rv_sig.csv", delimiter=",")
thar_rv_sigs = dd[:,1:14]


#Firstly, correct the Thorium-Argon RVs
nm = thar_rvs.shape[1]
nf_thar = thar_rvs.shape[0]
nf = rvs.shape[0]

thar_rvs_fitted = thar_rvs.copy()
rvs_corrected = rvs.copy()

#Fit to each file, in order to take out the rotation of the chip.
for i in range(nf_thar):
    x = np.arange(nm)
    p = np.polyfit(x,thar_rvs[i],1,w=thar_rv_sigs[i]**(-2.0))
    thar_rvs_fitted[i] = p[0]*x + p[1]
    
#Now go through the data and correct order by order
for i in range(nf):
    delta_t = np.abs(mjds[i] - thar_mjds)
    sorted_ix = np.argsort(delta_t)[0:nref]
    weights = delta_t[sorted_ix]**weight_exponent
    thar_ref = np.average(thar_rvs_fitted[sorted_ix], weights=weights,axis=0)
    rvs_corrected[i] = rvs[i] - thar_ref
    
#plt.clf()

rvs_med    = np.median(rvs_corrected[:,m_min:m_max+1],axis=1)
rvs_med_sig = np.std(rvs_corrected[:,m_min:m_max+1],axis=1)/np.sqrt(m_max-m_min)*1.2

weights = rv_sigs[:,m_min:m_max+1]**(-2.0)
rvs_mn    = np.average(rvs_corrected[:,m_min:m_max+1],axis=1,weights=weights)
rvs_mn_sig = np.sum(weights,axis=1)**(-0.5)


#plt.errorbar(mjds,rvs_med,rvs_med_sig,fmt='o')
plt.errorbar(mjds,rvs_mn,rvs_mn_sig,fmt='o')
plt.ylabel("Corrected RV (m/s)")
plt.xlabel("Date (MJD)")