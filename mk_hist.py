#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os import mkdir
from os.path import exists
import pickle
from scipy import signal
from scipy import optimize as opt

import numpy as np
import pandas as pd
import obspy

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from common_nipissis import root_dir, sensitivity_g, sensitivity_h, rms
import warnings
warnings.simplefilter("ignore")

LEN_FILE = 16000


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


def get_spectrum(traces, sample_rates):
    f, Pxx = signal.periodogram(traces[0],
                                sample_rates[0],
                                scaling='spectrum')
    data = np.empty((len(traces), Pxx.size))
    data[0] = Pxx
    for n in range(1, len(traces)):
        f, Pxx = signal.periodogram(traces[n],
                                    sample_rates[n],
                                    scaling='spectrum')
        data[n] = Pxx

    return f, data


def fwhm(y):
    def gauss(x, p):  # p[0]==mean, p[1]==stdev
        return 1.0/(p[1]*np.sqrt(2*np.pi))*np.exp(-(x-p[0])**2/(2*p[1]**2))*p[2]

    def errfunc(p, x, y):
        return gauss(x, p) - y  # Distance to the target function

    x = np.arange(len(y))
    y = y.copy()
    y = np.log10(y)
    y = y - min(y)

    # Fit a guassian
    p0 = [len(y)//2, len(y)//2, np.log10(100)]  # Inital guess
    p1, success = opt.leastsq(errfunc, p0, args=(x, y))

    fit_mu, fit_stdev, fit_scale = p1

    fwhm = 2*np.sqrt(2*np.log(2))*fit_stdev
    # plt.clf()
    # plt.scatter(x, y)
    # plt.plot(x, gauss(x, p1))
    # plt.show()
    return fwhm


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
max_rms_amplitudes = np.empty(70)
fwhm_time = np.empty(70)
for ntr in range(70):
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

    for i, file in enumerate(files):
        filename = root_dir+'all/'+file
        traces = obspy.read(filename)
        ntraces = len(traces)
        if i == 0:
            all_traces = np.zeros((len(files), ntraces, traces[0].data.size), np.float32)
            all_sample_rates = np.zeros((len(files), ntraces), np.float32)

        all_sample_rates[i, 0:len(traces)] = [
            tr.stats.sampling_rate for tr in traces
        ]

        if site == 0 or site == 1:  # Fosse de l'Est or Endicott
            if ntraces != 24:
                print('\nWarning: only '+str(ntraces)+' in '+filename+'\n')
            if sensor == 'Geophone':
                for nt in range(ntraces):
                    tr = traces[nt]
                    tr.data *= 1000.0 * tr.stats.calib / sensitivity_g[nt]
                    all_traces[i, nt, :] = tr.data
            else:
                for nt in range(ntraces):
                    tr = traces[nt]
                    tr.data *= tr.stats.calib / sensitivity_h[nt]
                    all_traces[i, nt, :] = tr.data
        else:
            if ntraces != 48:
                print('\nWarning: only '+str(ntraces)+' in '+filename+'\n')
            if sensor == 'Geophone':
                for nt in range(24):
                    tr = traces[nt]
                    tr.data *= 1000.0 * tr.stats.calib / sensitivity_g[nt]
                    all_traces[i, nt, :] = tr.data
            else:
                for nt in range(24):
                    tr = traces[nt+24]
                    tr.data *= tr.stats.calib / sensitivity_h[nt]
                    all_traces[i, nt, :] = tr.data

    all_traces = all_traces.reshape([-1, tr.data.size])
    all_sample_rates = all_sample_rates.flatten()
    mask = np.any(all_traces != 0, axis=1)
    all_traces, all_sample_rates = all_traces[mask], all_sample_rates[mask]

    fig, ax = plt.subplots(2, 2, figsize=[8.4, 6.8])
    ax[0, 0].hist(all_traces.flatten(), bins=30, log=True, histtype='stepfilled')
    if sensor == 'Geophone':
        ax[0, 0].set_xlabel('Particle Velocity (mm/s)')
    else:
        ax[0, 0].set_xlabel('Pressure (Pa)')
    ax[0, 0].set_ylabel('Count')

    rms_val = np.empty((all_traces.shape[0],))
    for nt in np.arange(all_traces.shape[0]):
        rms_val[nt] = rms(all_traces[nt, :])

    max_rms_amplitudes[ntr] = max(rms_val)

    assert (all_sample_rates == all_sample_rates[0]).all()
    # Sample rate in microseconds.
    # total_passage_time = (endtime-starttime).seconds / 60  # Minutes.
    # rms_val_temp = rms_val.reshape([len(files), -1])
    # rms_val_temp = np.mean(rms_val_temp, axis=1)
    # fwhm_time[ntr] = total_passage_time * fwhm(rms_val_temp) / len(rms_val_temp)
    # print(fwhm_time[ntr])

    ax[0, 1].hist(rms_val, bins=30, log=True, histtype='stepfilled')
    if sensor == 'Geophone':
        ax[0, 1].set_xlabel('RMS Trace Part. Vel. (mm/s)')
    else:
        ax[0, 1].set_xlabel('RMS Trace Pressure (Pa)')
    ax[0, 1].set_ylabel('Count')

    f, spectra = get_spectrum(all_traces, all_sample_rates)

    ax[1, 0].plot(f, spectra.mean(axis=0))
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_yscale('log')
    if sensor == 'Geophone':
        ax[1, 0].set_ylabel('Amplitude (mm/s RMS)')
    else:
        ax[1, 0].set_ylabel('Amplitude (Pa RMS)')
    ax[1, 0].set_xlabel('Frequency (Hz)')
    ax[1, 0].set_title('Spectre moyen')

    f_E_max = np.empty((spectra.shape[0],))
    for ns in np.arange(f_E_max.size):
        f_E_max[ns] = f[np.argmax(spectra[ns,:])]

    ax[1, 1].hist(f_E_max, bins=30)
    ax[1, 1].set_xlabel('Dominant Frequency (Hz)')
    ax[1, 1].set_ylabel('Count')

#    spectrum = spectra[np.argmax(rms_val)]
#    ax[1, 1].plot(f, spectrum)
#    ax[1, 1].set_xscale('log')
#    ax[1, 1].set_yscale('log')
#    if sensor == 'Geophone':
#        ax[1, 1].set_ylabel('Spectrum (mm/s RMS)')
#    else:
#        ax[1, 1].set_ylabel('Spectrum (Pa RMS)')
#    ax[1, 1].set_xlabel('Frequency (Hz)')
#    ax[1, 1].set_title('Fichier de plus grande amplitude RMS')

    fig.suptitle(sites[site]+', Train: '+train)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not exists('histograms'):
        mkdir('histograms')
    fig.savefig('histograms/train_no{0:02d}.pdf'.format(ntr))
    plt.clf()

    if ntr == 0:
        locator = mdates.AutoDateLocator(minticks=10, maxticks=20)
        formatter = mdates.ConciseDateFormatter(locator)

        mpl.rc('font', size=20)
        plt.figure(figsize=(12, 8))
        to_plot_rms = rms_val.reshape([-1, 24])
        to_plot_rms = to_plot_rms.mean(axis=1)
        time = pd.date_range(
            starttime,
            endtime,
            periods=len(to_plot_rms),
        )
        plt.plot_date(time, to_plot_rms, ms=6, c='k', ls='-')
        plt.xlabel("Temps")
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_locator(locator)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.ylabel("Amplitude RMS moyenne [mm/s]")
        plt.grid(True, which='major', color='k', alpha=.35)
        plt.grid(True, which='minor', linestyle='--', color='k', alpha=.1)
        plt.minorticks_on()
        plt.semilogy()
        plt.tight_layout()
        plt.savefig("fig/data.png")
        plt.show()

pickle.dump(max_rms_amplitudes, open("rms_val.pkl", "wb"))

plt.clf()
plt.hist(max_rms_amplitudes, bins=20)
plt.xticks(range(0, 275, 25))
plt.xlabel('RMS Trace Part. Vel. (mm/s)')
plt.ylabel('Count')
fig.savefig('histograms/max_amplitudes.png')
plt.show()

print(max(max_rms_amplitudes))
max_rms_amplitudes_temp = np.sort(max_rms_amplitudes)[:-1]
print(f"Minimum: {np.percentile(max_rms_amplitudes_temp, 0)}")
print(f"Percentile 33: {np.percentile(max_rms_amplitudes_temp, 33)}")
print(f"Percentile 66: {np.percentile(max_rms_amplitudes_temp, 66)}")
print(f"Maximum: {np.percentile(max_rms_amplitudes_temp, 100)}")

fwhm_temp = np.sort(fwhm_time)[:-2]
print(f"Min: {min(fwhm_temp)}")
print(f"Max: {max(fwhm_temp)}")
print(f"Mean: {np.mean(fwhm_temp)}")
print(f"Median: {np.median(fwhm_temp)}")
print(f"Std: {np.std(fwhm_temp)}")
plt.hist(fwhm_temp, bins=30)
plt.xlabel("Durée du passage [min]")
plt.ylabel("Count")
plt.savefig("histograms/time.png")
plt.show()


print(passage_times)
mask = passage_times["Train"].str.contains("_")
selected_trains = passage_times.loc[mask, ["passage_start", "passage_end"]]
passage_start = pd.to_numeric(pd.to_datetime(selected_trains["passage_start"]))
passage_end = pd.to_numeric(pd.to_datetime(selected_trains["passage_end"]))
mean_times = pd.to_datetime((passage_end+passage_start) / 2)
mean_times = mean_times.reset_index(drop=True)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(mean_times)
mean_times.to_csv("mean_times.csv", index=False)
