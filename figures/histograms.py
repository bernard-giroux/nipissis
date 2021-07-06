#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import warnings
from datetime import datetime

from scipy import signal
from scipy import optimize as opt
import numpy as np
import pandas as pd
import obspy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from common_nipissis import root_dir, sensitivity_g, rms
from catalog import catalog, Figure, Metadata

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


def gauss(x, p):
    mean, std, scale = p
    return scale / (std*np.sqrt(2*np.pi)) * np.exp(-(x-mean)**2/(2*std**2))


def errfunc(p, x, y):
    return gauss(x, p) - y  # Distance to the target function


def fwhm(y):
    x = np.arange(len(y))
    y = y.copy()
    y = np.log10(y)
    y = y - min(y)

    # Fit a Gaussian.
    p0 = [len(y)//2, len(y)//2, np.log10(100)]  # Inital guess.
    p1, success = opt.leastsq(errfunc, p0, args=(x, y))

    _, fit_std, _ = p1

    fwhm = 2 * np.sqrt(2*np.log(2)) * fit_std
    return fwhm


class Histograms(Metadata):
    def generate(self):
        site_rms = []
        for i in range(1, 4):
            with open(f'site{i}_rms.pkl', 'rb') as f:
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
                passage_times.loc[mask_match, 'passage_start'] = (
                    passage_site[mask_match] - delta
                )
                passage_times.loc[mask_match, 'passage_end'] = (
                    passage_site[mask_match] + delta
                )

        site = 0
        SENSOR = 'Geophone'
        QTY_TRAINS = 70
        max_rms_amplitudes = np.empty(QTY_TRAINS)
        fwhm_time = np.empty(QTY_TRAINS)
        for ntr in range(QTY_TRAINS):
            print(f"Processing train {ntr}.")
            train = f'_ ({ntr})'
            train_match = passage_times['Train'] == train
            passages = passage_times.loc[
                train_match,
                ['passage_start', 'passage_end'],
            ]
            starttime, endtime = passages.iloc[0]
            if ntr == 0:
                self["passage_0"] = [
                    starttime.timestamp(), endtime.timestamp(),
                ]

            files = get_file_list(starttime, endtime, site+1, SENSOR)
            if len(files) == 0:
                site += 1
                files = get_file_list(starttime, endtime, site+1, SENSOR)

            for i, file in enumerate(files):
                filename = root_dir+'all/'+file
                traces = obspy.read(filename)
                ntraces = len(traces)
                if i == 0:
                    all_traces = np.zeros(
                        (len(files), ntraces, traces[0].data.size),
                        np.float32,
                    )
                    all_sample_rates = np.zeros(
                        (len(files), ntraces),
                        np.float32,
                    )

                all_sample_rates[i, 0:len(traces)] = [
                    tr.stats.sampling_rate for tr in traces
                ]

                if site in [0, 1]:  # Fosse de l'Est or Endicott.
                    if ntraces != 24:
                        print(f"Warning: only {ntraces} traces in {filename}.")
                else:
                    if ntraces != 48:
                        print(f"Warning: only {ntraces} traces in {filename}.")
                    ntraces = 24
                for nt in range(ntraces):
                    tr = traces[nt]
                    tr.data *= 1000.0 * tr.stats.calib / sensitivity_g[nt]
                    all_traces[i, nt, :] = tr.data

            all_traces = all_traces.reshape([-1, tr.data.size])
            all_sample_rates = all_sample_rates.flatten()
            mask = np.any(all_traces != 0, axis=1)
            all_traces = all_traces[mask]
            all_sample_rates = all_sample_rates[mask]

            rms_val = np.empty((all_traces.shape[0],))
            for nt in np.arange(all_traces.shape[0]):
                rms_val[nt] = rms(all_traces[nt, :])

            if ntr == 0:
                self["rms_0"] = rms_val
            max_rms_amplitudes[ntr] = max(rms_val)

            assert (all_sample_rates == all_sample_rates[0]).all()

            total_passage_time = (endtime-starttime).seconds / 60  # Minutes.
            rms_val_temp = rms_val.reshape([len(files), -1])
            rms_val_temp = np.mean(rms_val_temp, axis=1)
            fwhm_time[ntr] = (
                total_passage_time * fwhm(rms_val_temp) / len(rms_val_temp)
            )

        self["max_rms"] = max_rms_amplitudes
        self["fwhm"] = fwhm_time


class SampleData(Figure):
    Metadata = Histograms

    def plot(self, data):
        locator = mdates.AutoDateLocator(minticks=1, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)

        rms_val = data["rms_0"]
        to_plot_rms = rms_val.reshape([-1, 24])
        to_plot_rms = to_plot_rms.mean(axis=1)

        starttime, endtime = data["passage_0"]
        starttime = datetime.fromtimestamp(starttime)
        endtime = datetime.fromtimestamp(endtime)

        time = pd.date_range(
            starttime,
            endtime,
            periods=len(to_plot_rms),
        )
        plt.figure(figsize=[4.33, 3])
        plt.plot_date(time, to_plot_rms, ms=2, c='k', ls='-')
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_locator(locator)
        plt.gca().xaxis.set_major_formatter(formatter)

        idx_max = np.argmax(to_plot_rms)
        x_max = time[idx_max]
        y_max = to_plot_rms[idx_max]
        plt.plot_date(
            [time[0], x_max],
            [y_max, y_max],
            ms=0,
            ls='--',
            c='k',
        )
        plt.text(time.min(), y_max, "Maximum", ha='left', va='top')
        DIFF_IDX = 20
        x = time[DIFF_IDX:DIFF_IDX+1].mean()
        y = to_plot_rms[DIFF_IDX:DIFF_IDX+1].mean()
        DT = 1 / 5
        plt.annotate(
            "8 seconds",
            [x, y*.95],
            [0, -16],
            ha='center',
            va='top',
            arrowprops={'arrowstyle': f'-[, widthB={DT}, lengthB=.2'},
            textcoords='offset points',
        )

        plt.xlabel("Time")
        plt.ylabel("Average RMS amplitude (mm/s)")
        plt.xlim([time.min(), time.max()])
        plt.grid(True, which='major', color='k', alpha=.35)
        plt.grid(True, which='minor', linestyle='--', color='k', alpha=.1)
        plt.minorticks_on()
        plt.semilogy()
        plt.tight_layout()


class HistogramAmplitudes(Figure):
    Metadata = Histograms

    def plot(self, data):
        plt.rcParams["font.family"] = "serif"
        max_rms_amplitudes = data["max_rms"]
        # Maximum 24-channel mean RMS amplitude [mm/s].
        # Mean over 8 minutes.
        # Maximum over train passage.
        print("AMPLITUDE STATISTICS")
        print(f"Minimum: {np.min(max_rms_amplitudes)}")
        print(f"Percentile 33: {np.percentile(max_rms_amplitudes, 33)}")
        print(f"Percentile 66: {np.percentile(max_rms_amplitudes, 66)}")
        print(f"Maximum: {np.max(max_rms_amplitudes)}")
        plt.hist(max_rms_amplitudes, bins=20, color=[.3]*3)
        plt.xticks(range(0, 275, 25))
        plt.xlabel("Vibration magnitude (mm/s)")
        plt.ylabel("Count")


class HistogramTimes(Figure):
    Metadata = Histograms

    def plot(self, data):
        plt.rcParams["font.family"] = "serif"
        fwhm_time = data["fwhm"]
        fwhm_temp = np.sort(fwhm_time)[:-1]
        print("PASSAGE TIME STATISTICS")
        print(f"Minimum: {min(fwhm_temp)}")
        print(f"Maximum: {max(fwhm_temp)}")
        print(f"Mean: {np.mean(fwhm_temp)}")
        print(f"Median: {np.median(fwhm_temp)}")
        print(f"Std: {np.std(fwhm_temp)}")
        plt.hist(fwhm_temp, bins=30, color=[.3]*3)
        plt.xlabel("Passage duration (min)")
        plt.ylabel("Count")


# print(passage_times)
# mask = passage_times["Train"].str.contains("_")
# selected_trains = passage_times.loc[mask, ["passage_start", "passage_end"]]
# passage_start = pd.to_numeric(pd.to_datetime(selected_trains["passage_start"]))
# passage_end = pd.to_numeric(pd.to_datetime(selected_trains["passage_end"]))
# mean_times = pd.to_datetime((passage_end+passage_start) / 2)
# mean_times = mean_times.reset_index(drop=True)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(mean_times)
# mean_times.to_csv("mean_times.csv", index=False)


catalog.register(SampleData)
catalog.register(HistogramAmplitudes)
catalog.register(HistogramTimes)
