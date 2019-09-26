#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 07:10:15 2019

@author: giroux
"""
import datetime
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from obspy import read
import warnings
warnings.simplefilter("ignore")

from common_nipissis import root_dir, sensitivity_g, rms


data_dir = root_dir+'all/'

# input voltage due to one shot = data point * DESCALING_FACTOR / STACK_COUNT
#
# STACK_COUNT is always 1 here

jours = (5, 6, 7, 8, 9)


for j in jours:

    starttime = []
    mE = []

    files = [f for f in glob.glob(data_dir+'g_2019-08-0'+str(j)+'*')]
    for f in sorted(files):
        print('Processing '+f)
        st = read(f)
        if len(st) != 24:
            print('\n\nWarning: only '+str(len(st))+' in file '+f+'\n\n')
        starttime.append(st[0].stats.starttime.matplotlib_date)
        tmp = np.empty((len(st),))
        for nt in range(len(st)):
            tr = st[nt]
            tmp[nt] = 1000 * rms(tr.data * tr.stats.calib / sensitivity_g[nt])
        mE.append(np.mean(tmp))

    plt.figure(figsize=(20, 5))
    plt.plot_date(starttime, mE, 'o')
    plt.gca().set_yscale('log')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    plt.grid()
    plt.title('Geophones 2019-08-0'+str(j), fontsize=18)
    plt.ylabel('24 channel mean RMS amplitude (mm/s)', fontsize=14)
    plt.xlabel('Time', fontsize=14)
    plt.xlim(datetime.datetime(2019, 8, j-1, 23, 50),
             datetime.datetime(2019, 8, j+1, 0, 10))
    plt.tight_layout()
    plt.savefig('fig/g_2019-08-0'+str(j)+'_mean_rms.pdf', bbox_inches='tight')
    plt.show(block=False)
