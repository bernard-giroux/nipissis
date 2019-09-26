#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 20:14:58 2019

@author: giroux
"""

import numpy as np
import matplotlib.pyplot as plt

from obspy import read

import warnings
warnings.simplefilter("ignore")

from common_nipissis import root_dir, sensitivity_g, sensitivity_h, rms

data_dir = root_dir+'geophones/20190806/'

# input voltage due to one shot = data point * DESCALING_FACTOR / STACK_COUNT
#
# STACK_COUNT is always 1 here



f_debut = 8500
f_fin = 8600

nf = f_fin - f_debut

mE = np.empty((nf, 24))
starttime = ['' for x in range(nf+1)]


for n in range(f_debut, f_fin):

    st = read(data_dir+str(n)+'.dat')
    for nt in range(24):
        tr = st[nt]
        if nt == 0:
            starttime[n-f_debut] = tr.stats.starttime.matplotlib_date
        mE[n-f_debut, nt] = 1000 * rms(tr.data * tr.stats.calib / sensitivity_g[nt])
starttime[-1] = tr.stats.endtime.matplotlib_date

fig = plt.figure()
ax = fig.add_subplot(111)
c = ax.pcolor(np.arange(0.5, 25), starttime, mE)
ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
ax.yaxis_date()
cbar = plt.colorbar(c, ax=ax)
cbar.ax.set_ylabel('Vitesse particule RMS (mm/s)')
plt.xlabel('Geophone no')
plt.ylabel('Jour & heure')
plt.tight_layout()

plt.show()


data_dir = root_dir+'hydrophones/20190806/'

# input voltage due to one shot = data point * DESCALING_FACTOR / STACK_COUNT
#
# STACK_COUNT is always 1 here



f_debut = 105
f_fin = 180

nf = f_fin - f_debut

mE = np.empty((nf, 24))
starttime = ['' for x in range(nf+1)]


for n in range(f_debut, f_fin):

    st = read(data_dir+str(n)+'.dat')
    for nt in range(24):
        tr = st[nt]
        if nt == 0:
            starttime[n-f_debut] = tr.stats.starttime.matplotlib_date
        mE[n-f_debut, nt] = rms(tr.data * tr.stats.calib / sensitivity_h[nt])
starttime[-1] = tr.stats.endtime.matplotlib_date

fig = plt.figure()
ax = fig.add_subplot(111)
c = ax.pcolor(np.arange(0.5, 25), starttime, mE)
ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
ax.yaxis_date()
cbar = plt.colorbar(c, ax=ax)
cbar.ax.set_ylabel('Amplitude RMS (Pa)')
plt.xlabel('Hydrophone no')
plt.ylabel('Jour & heure')
plt.tight_layout()

plt.show()
