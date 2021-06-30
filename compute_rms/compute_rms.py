#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import listdir
import numpy as np
import obspy

import warnings
warnings.simplefilter("ignore")

sensitivity_g = [23.4 for x in range(24)]


def rms(x):
    return np.sqrt(np.sum(x*x) / x.size)


files = listdir('.')
files.sort()
files = list(
    filter(
        lambda x: 'compute_rms' not in x and 'rms_values' not in x,
        files,
    )
)

for i, file in enumerate(files):
    filename = './' + file
    traces = obspy.read(filename)
    ntraces = len(traces)
    if i == 0:
        all_traces = np.zeros((len(files), ntraces, traces[0].data.size))
        all_sample_rates = np.zeros((len(files), ntraces))

    all_sample_rates[i, :ntraces] = [
        tr.stats.sampling_rate for tr in traces
    ]

    for nt in range(ntraces):
        tr = traces[nt]
        tr.data *= 1000.0 * tr.stats.calib / sensitivity_g[nt]
        all_traces[i, nt, :] = tr.data

rms_val = np.empty((all_traces.shape[0], all_traces.shape[1]))
for nt in range(all_traces.shape[0]):
    for nc in range(all_traces.shape[1]):
        rms_val[nt, nc] = rms(all_traces[nt, nc, :])

files = np.array(files)
rms_val = np.concatenate([files[:, None], rms_val[:, :]], axis=1)

np.savetxt('rms_values.txt', rms_val, fmt="%s")
