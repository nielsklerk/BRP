# Author B. Tabone August 2022
# This script runs plots a JWST spectrum with 'HCN','pH2O','oH2O','OH' ,'CO2','13CO2'
# Grey sepctrum is the total specrum and colored spec corresponds to various molecules

import numpy
from scipy.constants import c,h
from scipy.constants import k as k_b
import os,sys
sys.path.append('radex-version-IR_clean_v6/')
import radexpy.run_plot as radexpy
import pickle
from spectres import spectral_resampling_numba as spectres
import matplotlib.pyplot as mpl
from scipy.stats import norm
import scipy.constants as con
import numpy as np
import os

# Input parameters
N      = 1e16
T      = 600.
Rdisk  = 0.2 #au  
wmin   = 13.
wmax   = 17.



wlth,flux  = radexpy.disk_convo(species='CO2', T=T,N=N,DV=4.71,R=2000,Rdisk=Rdisk,distance=140.,wlthmin=wmin,wlthmax=wmax)
mpl.step(wlth,flux,where='mid',color='g')
flux_tot = flux

wlth,flux  = radexpy.disk_convo(species='H2O', T=T,N=N*30,DV=4.71,R=2000,Rdisk=Rdisk,distance=140.,wlthmin=wmin,wlthmax=wmax)
mpl.step(wlth,flux,where='mid',color='b')
flux_tot = flux_tot+flux

wlth,flux  = radexpy.disk_convo(species='OH', T=T,N=N,DV=4.71,R=2000,Rdisk=Rdisk,distance=140.,wlthmin=wmin,wlthmax=wmax)
mpl.step(wlth,flux,where='mid',color='r')
flux_tot = flux_tot+flux

wlth,flux  = radexpy.disk_convo(species='HCN', T=T,N=N,DV=4.71,R=2000,Rdisk=Rdisk,distance=140.,wlthmin=wmin,wlthmax=wmax)
mpl.step(wlth,flux,where='mid',color='orange')
flux_tot = flux_tot+flux

wlth,flux  = radexpy.disk_convo(species='C2H2', T=T,N=N,DV=4.71,R=2000,Rdisk=Rdisk,distance=140.,wlthmin=wmin,wlthmax=wmax)
mpl.step(wlth,flux,where='mid',color='purple')
flux_tot = flux_tot+flux

mpl.fill_between(wlth,flux_tot,step='mid',lw=1.5,color='k',zorder=0,alpha=0.5)




mpl.xlabel(r'$\lambda [\mu m]$ ')
mpl.ylabel(r'Flux [Jy]')
mpl.xlim((wmin,wmax))
mpl.xscale('linear')
mpl.show()

