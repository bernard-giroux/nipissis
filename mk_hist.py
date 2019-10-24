#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os import mkdir
from os.path import exists
import pickle

import numpy as np
import pandas as pd
import obspy

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from common_nipissis import root_dir, sensitivity_g, sensitivity_h, rms
import warnings
warnings.simplefilter("ignore")

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

class Site_rms:
    def __init__(self, starttime_g, mE_g, starttime_h, mE_h):
        self.mE_g = mE_g
        self.mE_h = mE_h
        self.starttime_g = [mdates.date2num(x) for x in starttime_g]
        self.starttime_h = [mdates.date2num(x) for x in starttime_h]


def get_file_list(starttime, endtime, site, sensor='Geophone'):

    if site == 1 or site == 2:  # Fosse de l'Est or Endicott
        if sensor == 'Geophone':
            prefix = 'g_2019-08-'
        else:
            prefix = 'h_2019-08-'
    else:
        prefix = 'gh_2019-08-'

    files = os.listdir(root_dir+'site'+str(site))
    files = np.array([f for f in files if prefix in f])
    times = [
        rreplace(
            rreplace(
                f.lstrip('gh_').split('.')[0], '-', ':', 1,
            ), '-', ':', 1,
        )
        for f in files
    ]
    times = [mdates.datestr2num(t) for t in times]
    times = [mdates.num2date(t).replace(tzinfo=None) for t in times]
    times = np.array(times)
    starttime = starttime.replace(tzinfo=None)
    endtime = endtime.replace(tzinfo=None)
    files = files[(starttime < times) & (times < endtime)]

    return files


sites = ('Site 1 - Fosse de l\'Est', 'Site 2 - Endicott', 'Site 3')

trains = pd.read_pickle('./train_data.pkl')
site_rms = []
with open('site1_rms.pkl', 'rb') as f:
    site_rms.append(pickle.load(f))
with open('site2_rms.pkl', 'rb') as f:
    site_rms.append(pickle.load(f))
with open('site3_rms.pkl', 'rb') as f:
    site_rms.append(pickle.load(f))

passage_times_files = os.listdir('./passage_times')
if passage_times_files:
    passage_times_files = [
        int(os.path.splitext(s)[0]) for s in passage_times_files
    ]
    passage_times = pd.read_pickle(
        f'./passage_times/{max(passage_times_files)}.pkl'
    )
else:
    passage_times = pd.read_pickle('./passage_times.pkl')
    for i, s in enumerate(site_rms):
        min_t = min(np.concatenate([s.starttime_g, s.starttime_h]))
        max_t = max(np.concatenate([s.starttime_g, s.starttime_h]))

        passage_site = passage_times[f'Site {i+1}']
        passage_site_m = [mdates.date2num(p) for p in passage_site]
        passage_site_m = np.array(passage_site_m)
        mask_match = (min_t < passage_site_m) & (passage_site_m < max_t)
        delta = pd.Timedelta(minutes=15)
        passage_times.loc[mask_match, 'passage_start'] = \
            passage_site[mask_match] - delta
        passage_times.loc[mask_match, 'passage_end'] = \
            passage_site[mask_match] + delta


site = 0
sensor = 'Geophone'
for ntr in range(71):
    train = '_ ('+str(ntr)+')'
    print('Je traite le train {0:02d}'.format(ntr))
    train_match = passage_times['Train'] == train
    passages = passage_times.loc[
        train_match,
        ['passage_start', 'passage_end'],
    ]
    [starttime, endtime] = passages.iloc[0]

    files = get_file_list(starttime, endtime, site+1, sensor)
    if len(files) == 0:
        site += 1
        files = get_file_list(starttime, endtime, site+1, sensor)

    all_traces = np.empty((0,),np.float32)
    for file in files:
        filename = root_dir+'site'+str(site+1)+'/'+file
        traces = obspy.read(filename)

        ntraces = len(traces)
        if site == 0 or site == 1:  # Fosse de l'Est or Endicott
            if ntraces != 24:
                print('\nWarning: only '+str(ntraces)+' in '+filename+'\n')
            if sensor == 'Geophone':
                for nt in range(ntraces):
                    tr = traces[nt]
                    tr.data *= 1000.0 * tr.stats.calib / sensitivity_g[nt]
                    all_traces = np.append(all_traces, tr.data)
            else:
                for nt in range(ntraces):
                    tr = traces[nt]
                    tr.data *= tr.stats.calib / sensitivity_h[nt]
                    all_traces = np.append(all_traces, tr.data)
        else:
            if ntraces != 48:
                print('\nWarning: only '+str(ntraces)+' in '+filename+'\n')
            if sensor == 'Geophone':
                for nt in range(24):
                    tr = traces[nt]
                    tr.data *= 1000.0 * tr.stats.calib / sensitivity_g[nt]
                    all_traces = np.append(all_traces, tr.data)
            else:
                for nt in range(24):
                    tr = traces[nt+24]
                    tr.data *= tr.stats.calib / sensitivity_h[nt]
                    all_traces = np.append(all_traces, tr.data)

    fig, ax = plt.subplots(1, 2, figsize=[8.4, 4.8])
    ax[0].hist(all_traces, bins=30, log=True)
    if sensor == 'Geophone':
        ax[0].set_xlabel('Particle Velocity (mm/s)')
    else:
        ax[0].set_xlabel('Pressure (Pa)')
    ax[0].set_ylabel('Count')

    all_traces = all_traces.reshape((-1, tr.data.size))
    rms_val = np.empty((all_traces.shape[0],))
    for nt in np.arange(all_traces.shape[0]):
        rms_val[nt] = rms(all_traces[nt,:])

    ax[1].hist(rms_val, bins=30, log=True)
    if sensor == 'Geophone':
        ax[1].set_xlabel('RMS Trace Part. Vel. (mm/s)')
    else:
        ax[1].set_xlabel('RMS Trace Pressure (Pa)')
    ax[1].set_ylabel('Count')

    fig.suptitle(sites[site]+', Train: '+train)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not exists('histograms'):
        mkdir('histograms')
    fig.savefig('histograms/train_no{0:02d}.pdf'.format(ntr))
