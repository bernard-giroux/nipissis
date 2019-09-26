#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 06:38:56 2019

@author: giroux
"""

import os
from shutil import copy2

from obspy import read
import warnings
warnings.simplefilter("ignore")
root_dir = '/Users/giroux/Desktop/nipissis/'

#jours = ('20190806', '20190807', '20190808')
jours = ('20190809',)

# Geophones

for j in jours:
    data_dir = root_dir+'geophones/'+j+'/'
    for filename in os.listdir(data_dir):
        print('Je traite '+filename)
        st = read(data_dir+filename)
        tr = st[0]
        starttime = str(tr.stats.starttime)
        s = starttime.replace('.000000Z', '.0Z').replace(':', '-')
        copy2(data_dir+filename, root_dir+'all/g_'+s+'.dat')

# Hydrophones

for j in jours:
    data_dir = root_dir+'hydrophones/'+j+'/'
    for filename in os.listdir(data_dir):
        print('Je traite '+filename)
        st = read(data_dir+filename)
        tr = st[0]
        starttime = str(tr.stats.starttime)
        s = starttime.replace('.000000Z', '.0Z').replace(':', '-')
        copy2(data_dir+filename, root_dir+'all/h_'+s+'.dat')


# Geophones & hydrophones
        
jours = ('20190811', '20190812')

for j in jours:
    data_dir = root_dir+'geo_hydro/'+j+'/'
    for filename in os.listdir(data_dir):
        print('Je traite '+filename)
        st = read(data_dir+filename)
        tr = st[0]
        starttime = str(tr.stats.starttime)
        s = starttime.replace('.000000Z', '.0Z').replace(':', '-')
        copy2(data_dir+filename, root_dir+'all/gh_'+s+'.dat')
