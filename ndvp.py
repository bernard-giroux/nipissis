#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import datetime
import glob
import os
import pickle
import sys
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib
import obspy
from itertools import chain

from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QVBoxLayout, QCheckBox, QMainWindow, QApplication, QGroupBox, QLabel,
    QLineEdit, QComboBox, QGridLayout, QSizePolicy, QFrame, QMessageBox,
    QListWidget, QPushButton
)
from PyQt5.QtCore import Qt, QLocale

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as \
    FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as \
    NavigationToolbar

import matplotlib.dates as mdates
from matplotlib.figure import Figure

from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt


from common_nipissis import root_dir, sensitivity_g, sensitivity_h, integrate, rms
import warnings
warnings.simplefilter("ignore")

register_matplotlib_converters()

# Make sure that we are using QT5
matplotlib.use('Qt5Agg')

locale = QLocale()


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


class Site_rms:
    def __init__(self, starttime_g, mE_g, starttime_h, mE_h):
        self.mE_g = mE_g
        self.mE_h = mE_h
        self.starttime_g = [mdates.date2num(x) for x in starttime_g]
        self.starttime_h = [mdates.date2num(x) for x in starttime_h]


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    # from http://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html

    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class Site_rms_canvas(MyMplCanvas):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axe1 = self.fig.add_subplot(111)
        self.axe2 = self.axe1.twinx()
        self.l1 = None     # line of geophone data
        self.l2 = None     # line of hydrophone data
        self.fills = []
        self.texts = []

    def plot(self, x1, y1, x2, y2, passage_times, current_train):

        if self.l1 is None:
            self.l1, = self.axe1.plot_date(x1, y1, 'o', c='C0')
            self.l2, = self.axe2.plot_date(x2, y2, '*', c='C1')
            self.axe1.set_yscale('log')
            self.axe2.set_yscale('log')
            self.axe1.set_ylabel('Mean RMS amplitude (mm/s)', color='C0')
            self.axe2.set_ylabel('Mean RMS amplitude (Pa)', color='C1')
            self.axe1.tick_params('y', colors='C0')
            self.axe2.tick_params('y', colors='C1')
            self.axe1.set_xlabel('Time')
            self.axe1.legend((self.l1, self.l2), ('geophones', 'hydrophones'),
                             bbox_to_anchor=(0., 1.02, 1., .102),
                             loc='lower left', ncol=2)
            self.fig.tight_layout()
        else:
            self.l1.set_data(x1, y1)
            self.l2.set_data(x2, y2)
            self.axe1.relim()
            self.axe1.autoscale_view()
            self.axe2.relim()
            self.axe2.autoscale_view()

        for item in chain(self.fills, self.texts):
            item.remove()
        self.fills, self.texts = [], []

        min_t = min(np.concatenate([x1, x2]))
        max_t = max(np.concatenate([x1, x2]))
        min_amp = min(np.concatenate([y1, y2]))
        max_amp = max(np.concatenate([y1, y2]))
        for train, start, end in zip(
                    passage_times['Train'],
                    passage_times['passage_start'],
                    passage_times['passage_end'],
                ):
            must_be_plotted = (
                not pd.isnull(start)
                and not pd.isnull(end)
                and start != 'nan'
                and end != 'nan'
                and (
                    min_t < mdates.date2num(start) < max_t
                    or min_t < mdates.date2num(end) < max_t
                )
            )
            if must_be_plotted:
                if train == current_train:
                    color = 'g'
                else:
                    color = 'r'
                alpha = .05 if '_' in train else .2
                fill = self.axe1.fill_betweenx(
                    [min_amp/1000, max_amp*1000],
                    start,
                    end,
                    step='mid',
                    color=color,
                    alpha=alpha,
                )
                self.fills.append(fill)
                text = self.axe1.text(
                    start,
                    max_amp / 10,
                    train,
                    rotation=90,
                    horizontalalignment='right',
                    verticalalignment='bottom',
                )
                self.texts.append(text)
        self.draw()


class Data_plot_canvas(MyMplCanvas):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axe1 = self.fig.add_subplot(121)
        self.axeb = self.fig.add_subplot(122)
        l, b, w, h = self.axe1.get_position().bounds
        ll, bb, ww, hh = self.axeb.get_position().bounds
        self.axe1.set_position([0.06, 0.15, 0.85, 0.77])
        self.axeb.set_position([0.925, 0.15, 0.02, 0.77])
        self.l1 = None
        self.i1 = None
        self.cbar = None

    def plot_trace(self, trace, sensor):
        self.axeb.set_visible(False)
        t = trace.stats.delta * np.arange(trace.stats.npts)
        if self.l1 is None:
            self.axe1.clear()
            self.l1, = self.axe1.plot(t, trace.data)
            self.axe1.set_xscale('linear')
            self.axe1.set_yscale('linear')
            self.axe1.grid()
        else:
            self.l1.set_data(t, trace.data)
            self.axe1.set_xscale('linear')
            self.axe1.set_yscale('linear')
            self.axe1.relim()
            self.axe1.autoscale_view()
        if sensor == 'Geophone':
            self.axe1.set_ylabel('Amplitude (mm/s)')
        else:
            self.axe1.set_ylabel('Amplitude (Pa)')
        self.axe1.set_xlabel('Time (s)')
        self.axe1.set_position([0.06, 0.15, 0.9, 0.77])
        self.axeb.set_position([0.925, 0.15, 0.02, 0.77])
        self.draw()
        self.i1 = None

    def plot_spectrum(self, trace, sensor):
        self.axeb.set_visible(False)
        f, Pxx = signal.periodogram(trace.data, trace.stats.sampling_rate,
                                    scaling='spectrum')
        if self.l1 is None:
            self.axe1.clear()
            self.l1, = self.axe1.plot(f, np.sqrt(Pxx))
            self.axe1.set_xscale('log')
            self.axe1.set_yscale('log')
        else:
            self.l1.set_data(f, np.sqrt(Pxx))
            self.axe1.set_xscale('log')
            self.axe1.set_yscale('log')
            self.axe1.relim()
            self.axe1.autoscale_view()
        if sensor == 'Geophone':
            self.axe1.set_ylabel('Spectrum (mm/s RMS)')
        else:
            self.axe1.set_ylabel('Spectrum (Pa RMS)')
        self.axe1.set_xlabel('Frequency (Hz)')
        self.axe1.set_position([0.06, 0.15, 0.9, 0.77])
        self.axeb.set_position([0.925, 0.15, 0.02, 0.77])
        self.draw()
        self.i1 = None

    def plot_displacement(self, trace, sensor):
        if sensor == 'Geophone':
            y = integrate(trace.data, trace.stats.sampling_rate)
        else:
            y = np.zeros(trace.data.shape)
        t = trace.stats.delta * np.arange(trace.stats.npts)
        self.axeb.set_visible(False)
        if self.l1 is None:
            self.axe1.clear()
            self.l1, = self.axe1.plot(t, y)
            self.axe1.set_xscale('linear')
            self.axe1.set_yscale('linear')
            self.axe1.grid()
        else:
            self.l1.set_data(t, y)
            self.axe1.set_xscale('linear')
            self.axe1.set_yscale('linear')
            self.axe1.relim()
            self.axe1.autoscale_view()
        if sensor == 'Geophone':
            self.axe1.set_ylabel('Amplitude (mm)')
        else:
            self.axe1.set_ylabel('')
        self.axe1.set_xlabel('Time (s)')
        self.axe1.set_position([0.06, 0.15, 0.9, 0.77])
        self.axeb.set_position([0.925, 0.15, 0.02, 0.77])
        self.draw()
        self.i1 = None

    def plot_traces(self, traces, sensor):
        t = traces[0].stats.delta * np.arange(traces[0].stats.npts)

        data = np.empty((24, traces[0].stats.npts))
        for n in range(24):
            data[n, :] = traces[n].data
        if self.i1 is None:
            self.axe1.clear()
        self.i1 = self.axe1.imshow(data, aspect='auto',
                                   extent=(t[0], t[-1], 24, 1))
        if self.cbar is None:
            self.cbar = self.fig.colorbar(self.i1, ax=self.axe1,
                                          cax=self.axeb,
                                          use_gridspec=True)
        else:
            self.cbar.on_mappable_changed(self.i1)
        self.axe1.set_xlabel('Time (s)')
        self.axe1.set_ylabel('Trace no')
        self.axe1.set_xscale('linear')
        if sensor == 'Geophone':
            self.cbar.set_label('Amplitude (mm/s)')
        else:
            self.cbar.set_label('Amplitude (Pa)')
        self.axeb.set_visible(True)
        self.axe1.set_position([0.06, 0.15, 0.85, 0.77])
        self.axeb.set_position([0.925, 0.15, 0.02, 0.77])
        self.draw()
        self.l1 = None

    def plot_spectra(self, traces, sensor):
        f, Pxx = signal.periodogram(traces[0].data,
                                    traces[0].stats.sampling_rate,
                                    scaling='spectrum')
        data = np.empty((24, Pxx.size))
        data[0, :] = np.log10(Pxx)
        for n in range(1, 24):
            f, Pxx = signal.periodogram(traces[n].data,
                                        traces[n].stats.sampling_rate,
                                        scaling='spectrum')
            data[n, :] = np.log10(Pxx)

        if self.i1 is None:
            self.axe1.clear()
        self.i1 = self.axe1.imshow(data, aspect='auto',
                                   extent=(f[0], f[-1], 24, 1))
        if self.cbar is None:
            self.cbar = self.fig.colorbar(self.i1, ax=self.axe1,
                                          cax=self.axeb,
                                          use_gridspec=True)
        else:
            self.cbar.on_mappable_changed(self.i1)
        self.axe1.set_xlabel('Frequency (Hz)')
        self.axe1.set_ylabel('Trace no')
        self.axe1.set_xscale('log')
        if sensor == 'Geophone':
            self.cbar.set_label('log spectrum (mm/s RMS)')
        else:
            self.cbar.set_label('log spectrum (Pa RMS)')
        self.axeb.set_visible(True)
        self.axe1.set_position([0.06, 0.15, 0.85, 0.77])
        self.axeb.set_position([0.925, 0.15, 0.02, 0.77])
        self.draw()
        self.l1 = None

    def plot_displacements(self, traces, sensor):
        t = traces[0].stats.delta * np.arange(traces[0].stats.npts)
        data = np.zeros((24, traces[0].stats.npts))
        if sensor == 'Geophone':
            for n in np.arange(24):
                data[n,:] = integrate(traces[n].data, traces[n].stats.sampling_rate)

        if self.i1 is None:
            self.axe1.clear()
        self.i1 = self.axe1.imshow(data, aspect='auto',
                                   extent=(t[0], t[-1], 24, 1))
        if self.cbar is None:
            self.cbar = self.fig.colorbar(self.i1, ax=self.axe1,
                                          cax=self.axeb,
                                          use_gridspec=True)
        else:
            self.cbar.on_mappable_changed(self.i1)
        self.axe1.set_xlabel('Time (s)')
        self.axe1.set_ylabel('Trace no')
        self.axe1.set_xscale('linear')
        if sensor == 'Geophone':
            self.cbar.set_label('Amplitude (mm)')
        else:
            self.cbar.set_label('')
        self.axeb.set_visible(True)
        self.axe1.set_position([0.06, 0.15, 0.85, 0.77])
        self.axeb.set_position([0.925, 0.15, 0.02, 0.77])
        self.draw()
        self.l1 = None
#
# Main class
#
class PyNDVP(QMainWindow):

    def __init__(self):
        super().__init__()

        self.site_rms = []
        self.traces = None
        self.load_data()
        self.init_ui()
        self.change_train()
        self.manage_dummy_trains()

        s = self.site_rms[self.site_no.currentIndex()]
        starttime = mdates.num2date(s.starttime_g[0])
        self.startday.setText(str(starttime.day))
        self.starthour.setText(str(starttime.hour))
        self.startminute.setText(str(starttime.minute))
        endtime = mdates.num2date(s.starttime_g[-1])
        endtime = endtime.replace(minute=endtime.minute+1)

        self.endday.setText(str(endtime.day))
        self.endhour.setText(str(endtime.hour))
        self.endminute.setText(str(endtime.minute))

        files = self.get_file_list(starttime, endtime)
        self.data_file.addItems(files)
        self.get_traces()
        self.data_plot.plot_trace(self.traces[self.channels.currentRow()],
                                  self.type_sensor.currentText())

        self.analyze_timedeltas()

    def init_ui(self):

        self.site_no = QComboBox()
        self.site_no.addItem('Site 1 - Fosse de l\'Est')
        self.site_no.addItem('Site 2 - Endicott')
        self.site_no.addItem('Site 3')
        self.site_no.currentIndexChanged.connect(self.change_site)

        self.train_list = QComboBox()
        for train in self.passage_times.loc[:, 'Train']:
            self.train_list.addItem(train)
        self.train_list.currentTextChanged.connect(
            lambda: (self.change_train(), self.start_end_changed())
        )

        self.passage_start = QLineEdit()
        self.passage_start.setText('')
        self.passage_start.setMaximumWidth(300)
        self.passage_start.editingFinished.connect(
            lambda: self.modify_passage_time(
                'passage_start',
                self.train_list.currentText(),
                self.passage_start.text(),
            )
        )

        self.passage_end = QLineEdit()
        self.passage_end.setText('')
        self.passage_end.setMaximumWidth(300)
        self.passage_start.editingFinished.connect(
            lambda: self.modify_passage_time(
                'passage_start',
                self.train_list.currentText(),
                self.passage_start.text(),
            )
        )

        self.save_button = QPushButton()
        self.save_button.setText("Save passage times")
        self.save_button.clicked.connect(self.save_passage_times)

        self.hist_button = QPushButton()
        self.hist_button.setText('Histograms')
        self.hist_button.clicked.connect(self.get_histograms)

        self.rms_plot = Site_rms_canvas()
        self.toolbar = NavigationToolbar(self.rms_plot, self)
        self.rms_plot.mpl_connect('button_press_event', self.handle_click)

        self.startday = QLineEdit()
        self.startday.setValidator(QIntValidator())
        self.startday.setMaximumWidth(40)
        self.startday.editingFinished.connect(self.start_end_changed)
        self.starthour = QLineEdit()
        self.starthour.setValidator(QIntValidator())
        self.starthour.setMaximumWidth(40)
        self.starthour.editingFinished.connect(self.start_end_changed)
        self.startminute = QLineEdit()
        self.startminute.setValidator(QIntValidator())
        self.startminute.setMaximumWidth(40)
        self.startminute.editingFinished.connect(self.start_end_changed)
        self.endday = QLineEdit()
        self.endday.setValidator(QIntValidator())
        self.endday.setMaximumWidth(40)
        self.endday.editingFinished.connect(self.start_end_changed)
        self.endhour = QLineEdit()
        self.endhour.setValidator(QIntValidator())
        self.endhour.setMaximumWidth(40)
        self.endhour.editingFinished.connect(self.start_end_changed)
        self.endminute = QLineEdit()
        self.endminute.setValidator(QIntValidator())
        self.endminute.setMaximumWidth(40)
        self.endminute.editingFinished.connect(self.start_end_changed)

        self.data_file = QComboBox()
        self.data_file.setMinimumWidth(250)
        self.data_file.currentIndexChanged.connect(self.file_changed)

        self.filter_by_train = QCheckBox()
        self.filter_by_train.stateChanged.connect(self.start_end_changed)

        self.channels = QListWidget()
        for n in range(24):
            self.channels.addItem(str(n+1))
        # self.channels.setSelectionMode(QAbstractItemView.MultiSelection)
        self.channels.itemSelectionChanged.connect(self.update_data_plot)
        self.channels.setMaximumWidth(40)
        self.channels.setCurrentRow(0)
        self.all_channels = QCheckBox('Show All Channels', self)
        self.all_channels.stateChanged.connect(self.update_data_plot)
        self.type_plot = QComboBox()
        self.type_plot.addItem('Trace')
        self.type_plot.addItem('Spectrum')
        self.type_plot.addItem('Displacement')
        self.type_plot.currentIndexChanged.connect(self.update_data_plot)
        self.type_sensor = QComboBox()
        self.type_sensor.addItem('Geophone')
        self.type_sensor.addItem('Hydrophone')
        self.type_sensor.currentIndexChanged.connect(self.sensor_changed)

        agb = QGroupBox('Data Selection')
        gl = QGridLayout()
        label = QLabel('Start day')
        label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        gl.addWidget(label, 0, 0)
        gl.addWidget(self.startday, 0, 1)
        label = QLabel('Start Hour')
        label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        gl.addWidget(label, 0, 2)
        gl.addWidget(self.starthour, 0, 3)
        label = QLabel('Start Minute')
        label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        gl.addWidget(label, 0, 4)
        gl.addWidget(self.startminute, 0, 5)
        label = QLabel('End day')
        label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        gl.addWidget(label, 1, 0)
        gl.addWidget(self.endday, 1, 1)
        label = QLabel('End Hour')
        label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        gl.addWidget(label, 1, 2)
        gl.addWidget(self.endhour, 1, 3)
        label = QLabel('End Minute')
        label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        gl.addWidget(label, 1, 4)
        gl.addWidget(self.endminute, 1, 5)
        label = QLabel('File')
        label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        gl.addWidget(label, 0, 6)
        gl.addWidget(self.data_file, 0, 7)
        label = QLabel('Filter by selected train')
        label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        gl.addWidget(label, 1, 7)
        gl.addWidget(self.filter_by_train, 1, 6)
        label = QLabel('Channel')
        label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        gl.addWidget(label, 0, 8)
        gl.addWidget(self.channels, 0, 9, 2, 1)
        gl.addWidget(self.all_channels, 1, 10, 1, 2)
        label = QLabel('Plot')
        label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        gl.addWidget(label, 0, 10)
        gl.addWidget(self.type_plot, 0, 11)
        label = QLabel('Sensor Type')
        label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        gl.addWidget(label, 0, 12)
        gl.addWidget(self.type_sensor, 0, 13)
        agb.setLayout(gl)

        rgb = QGroupBox('Results')

        self.data_plot = Data_plot_canvas()
        toolbar2 = NavigationToolbar(self.data_plot, self)
        vbox = QVBoxLayout()
        vbox.addWidget(toolbar2)
        vbox.addWidget(self.data_plot)
        rgb.setLayout(vbox)

        mw = QFrame()
        self.setCentralWidget(mw)

        gl = QGridLayout()
        gl.addWidget(self.site_no, 0, 0)
        label = QLabel('Train')
        label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        gl.addWidget(label, 0, 1)
        gl.addWidget(self.train_list, 0, 2)
        label = QLabel('Passage start')
        label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        gl.addWidget(label, 0, 3)
        gl.addWidget(self.passage_start, 0, 4)
        label = QLabel('Passage end')
        label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        gl.addWidget(label, 0, 5)
        gl.addWidget(self.passage_end, 0, 6)
        gl.addWidget(self.hist_button, 0, 7)
        gl.addWidget(self.save_button, 0, 8)
        gl.addWidget(self.toolbar, 1, 0, 1, 9)
        gl.addWidget(self.rms_plot, 2, 0, 1, 9)
        gl.addWidget(agb, 3, 0, 1, 9)
        gl.addWidget(rgb, 4, 0, 1, 9)

        mw.setLayout(gl)

        self.setGeometry(50, 50, 1700, 1000)
        self.setWindowTitle('Nipissis Data Viewing & Processing')
        self.show()

    def load_data(self):
        self.trains = pd.read_pickle('./train_data.pkl')
        with open('site1_rms.pkl', 'rb') as f:
            self.site_rms.append(pickle.load(f))
        with open('site2_rms.pkl', 'rb') as f:
            self.site_rms.append(pickle.load(f))
        with open('site3_rms.pkl', 'rb') as f:
            self.site_rms.append(pickle.load(f))
        self.passage_times = self.load_passage_times()

    def load_passage_times(self):
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
            for i, s in enumerate(self.site_rms):
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

        return passage_times

    def save_passage_times(self):
        self.manage_dummy_trains(remove_empty=True)
        passage_times_files = [
            int(os.path.splitext(s)[0]) for s in os.listdir('./passage_times')
        ]
        i = max(passage_times_files)+1 if passage_times_files else 0
        self.passage_times.to_pickle(
            f'./passage_times/{i}.pkl'
        )

    def change_site(self):
        self.change_train()

        s = self.site_rms[self.site_no.currentIndex()]
        starttime = mdates.num2date(s.starttime_g[0])
        self.startday.setText(str(starttime.day))
        self.starthour.setText(str(starttime.hour))
        self.startminute.setText(str(starttime.minute))
        endtime = mdates.num2date(s.starttime_g[-1])
        endtime = endtime.replace(minute=endtime.minute+1)

#        endtime = starttime + datetime.timedelta(minutes=10)
        self.endday.setText(str(endtime.day))
        self.endhour.setText(str(endtime.hour))
        self.endminute.setText(str(endtime.minute))

        files = self.get_file_list(starttime, endtime)
        self.data_file.clear()
        self.data_file.addItems(files)
        self.get_traces()
        self.update_data_plot()

    def change_train(self):
        current_train = self.train_list.currentText()
        s = self.site_rms[self.site_no.currentIndex()]
        self.rms_plot.plot(
            s.starttime_g,
            s.mE_g,
            s.starttime_h,
            s.mE_h,
            self.passage_times,
            current_train,
        )
        train_match = self.passage_times['Train'].str.strip() == current_train
        passages = (
            self.passage_times.loc[
                train_match,
                ['passage_start', 'passage_end']
            ]
        )
        [[start, end]] = passages.values
        start, end = str(start), str(end)
        if '.' in start:
            start = start.split('.')[0]
        if '+' in start:
            start = start.split('+')[0]
        if '.' in end:
            end = end.split('.')[0]
        if '+' in start:
            start = start.split('+')[0]
        self.passage_start.setText(start)
        self.passage_end.setText(end)

    def get_file_list(self, starttime, endtime):

        site = self.site_no.currentIndex()+1
        sensor = self.type_sensor.currentText()

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
            self.rreplace(
                self.rreplace(
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

    def rreplace(self, s, old, new, occurrence):
        li = s.rsplit(old, occurrence)
        return new.join(li)

    def get_traces(self):
        site = self.site_no.currentIndex()+1
        filename = root_dir+'site'+str(site)+'/'+self.data_file.currentText()
        try:
            self.traces = obspy.read(filename)
        except OSError:
            # get_traces may be called before the list of files is updated
            return
        ntraces = len(self.traces)
        if site == 1 or site == 2:  # Fosse de l'Est or Endicott
            if ntraces != 24:
                print('\n\nWarning: only '+str(ntraces)+' in '+filename+'\n')
            if self.type_sensor.currentText() == 'Geophone':
                for nt in range(ntraces):
                    tr = self.traces[nt]
                    tr.data *= 1000.0 * tr.stats.calib / sensitivity_g[nt]
            else:
                for nt in range(ntraces):
                    tr = self.traces[nt]
                    tr.data *= tr.stats.calib / sensitivity_h[nt]
        else:
            if ntraces != 48:
                print('\n\nWarning: only '+str(ntraces)+' in '+filename+'\n')
            if self.type_sensor.currentText() == 'Geophone':
                for nt in range(24):
                    tr = self.traces[nt]
                    tr.data *= 1000.0 * tr.stats.calib / sensitivity_g[nt]
                self.traces = self.traces[:24]
            else:
                for nt in range(24):
                    tr = self.traces[nt+24]
                    tr.data *= tr.stats.calib / sensitivity_h[nt]
                self.traces = self.traces[24:]

    def update_data_plot(self):
        if self.traces is None:
            return
        if self.all_channels.isChecked():
            sensor = self.type_sensor.currentText()
            if self.type_plot.currentIndex() == 0:  # trace
                self.data_plot.plot_traces(self.traces, sensor)
            elif self.type_plot.currentIndex() == 1:
                self.data_plot.plot_spectra(self.traces, sensor)
            else:
                self.data_plot.plot_displacements(self.traces, sensor)
        else:
            tr_no = self.channels.currentRow()
            sensor = self.type_sensor.currentText()
            if self.type_plot.currentIndex() == 0:  # trace
                self.data_plot.plot_trace(self.traces[tr_no], sensor)
            elif self.type_plot.currentIndex() == 1:
                self.data_plot.plot_spectrum(self.traces[tr_no], sensor)
            else:
                self.data_plot.plot_displacement(self.traces[tr_no], sensor)

    def sensor_changed(self):
        starttime, endtime = self.fetch_start_end()
        files = self.get_file_list(starttime, endtime)
        if len(files) == 0:
            QMessageBox.warning(self, 'Warning',
                                'No file found for selected period of time')
            return
        self.data_file.clear()
        self.data_file.addItems(files)
        self.get_traces()
        self.update_data_plot()

    def start_end_changed(self):
        starttime, endtime = self.fetch_start_end()
        if starttime >= endtime:
            QMessageBox.warning(self, 'Warning',
                                'End time is earlier that start time')
            return
        files = self.get_file_list(starttime, endtime)
        if len(files) == 0:
            QMessageBox.warning(self, 'Warning',
                                'No file found for selected period of time')
            return
        files.sort()
        self.data_file.clear()
        self.data_file.addItems(files)
        self.get_traces()
        self.update_data_plot()

    def fetch_start_end(self, by_train=False):
        if self.filter_by_train.checkState() or by_train:
            train = self.train_list.currentText()
            train_match = self.passage_times['Train'] == train
            passages = self.passage_times.loc[
                train_match,
                ['passage_start', 'passage_end'],
            ]
            [starttime, endtime] = passages.iloc[0]
        else:
            starttime = datetime.datetime(2019, 8, int(self.startday.text()),
                                          int(self.starthour.text()),
                                          int(self.startminute.text()))
            endtime = datetime.datetime(2019, 8, int(self.endday.text()),
                                        int(self.endhour.text()),
                                        int(self.endminute.text()))
        starttime = starttime.replace(tzinfo=None)
        endtime = endtime.replace(tzinfo=None)

        return starttime, endtime

    def file_changed(self):
        self.get_traces()
        self.update_data_plot()

    def handle_click(self, event):
        if self.toolbar.mode:
            return

        new_value = event.xdata
        train = self.train_list.currentText()
        if event.button is matplotlib.backend_bases.MouseButton.LEFT:
            self.modify_passage_time('passage_start', train, new_value)
        elif event.button is matplotlib.backend_bases.MouseButton.RIGHT:
            self.modify_passage_time('passage_end', train, new_value)

        self.start_end_changed()

    def modify_passage_time(self, column, train, new_value):
        if isinstance(new_value, str):
            new_value = mdates.datestr2num(new_value)
        if isinstance(new_value, float):
            new_value = mdates.num2date(new_value)
        train_match = self.passage_times['Train'] == train
        self.passage_times.loc[train_match, column] = new_value

        self.change_train()
        self.manage_dummy_trains()

    def manage_dummy_trains(self, data=None, remove_empty=False):
        if data is None:
            in_place = True
            data = self.passage_times
        else:
            in_place = False

        trains = data['Train']
        dummies = trains[trains.str.contains('_')]
        next_idx = len(dummies)

        passages = data.loc[
            data['Train'] == f'_ ({next_idx-1})',
            ['passage_start', 'passage_end'],
        ]
        if len(passages) == 0:
            start, end = 123, 123  # Allow creating a new dummy.
        else:
            [[start, end]] = passages.values
        if pd.isnull(start) or pd.isnull(end) or start == 'nan' or end == 'nan':
            if remove_empty:
                data = data.loc[
                    data['Train'] != f'_ ({next_idx-1})'
                ]
        elif not remove_empty:
            data = data.append(
                pd.DataFrame(
                    [[f'_ ({next_idx})', pd.NaT, pd.NaT]],
                    columns=['Train', 'passage_start', 'passage_end'],
                )
            )

        self.update_train_list(data['Train'])

        if in_place:
            self.passage_times = data
        else:
            return data

    def update_train_list(self, train_list):
        current_trains = [
            self.train_list.itemText(i) for i in range(self.train_list.count())
        ]
        for train in train_list:
            if train not in current_trains:
                self.train_list.addItem(train)

    def get_histograms(self):
        starttime, endtime = self.fetch_start_end(by_train=True)
        files = self.get_file_list(starttime, endtime)
        if len(files) == 0:
            QMessageBox.warning(self, 'Warning',
                                'No file found for selected period of time')
            return

        # load in all data
        site = self.site_no.currentIndex()+1
        for i, file in enumerate(files):
            filename = root_dir+'site'+str(site)+'/'+file
            traces = obspy.read(filename)
            ntraces = len(traces)
            if i == 0:
                all_traces = np.zeros((len(files), ntraces, traces[0].data.size), np.float32)
                all_sample_rates = np.zeros((len(files), ntraces), np.float32)
            
            all_sample_rates[i, 0:len(traces)] = [
                tr.stats.sampling_rate for tr in traces
            ]

            if site == 1 or site == 2:  # Fosse de l'Est or Endicott
                if ntraces != 24:
                    print('\n\nWarning: only '+str(ntraces)+' in '+filename+'\n')
                if self.type_sensor.currentText() == 'Geophone':
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
                    print('\n\nWarning: only '+str(ntraces)+' in '+filename+'\n')
                if self.type_sensor.currentText() == 'Geophone':
                    for nt in range(24):
                        tr = traces[nt]
                        tr.data *= 1000.0 * tr.stats.calib / sensitivity_g[nt]
                        all_traces[i, nt, :] = tr.data
                else:
                    for nt in range(24):
                        tr = traces[nt+24]
                        tr.data *= tr.stats.calib / sensitivity_h[nt]
                        all_traces[i, nt, :] = tr.data

        all_traces = all_traces.reshape((-1, tr.data.size))
        all_sample_rates = all_sample_rates.flatten()
        mask = np.any(all_traces != 0, axis=1)
        all_traces, all_sample_rates = all_traces[mask], all_sample_rates[mask]

        fig, ax = plt.subplots(2, 2, figsize=[8.4, 6.8])
        ax[0, 0].hist(all_traces.flatten(), bins=30, log=True)
        if self.type_sensor.currentText() == 'Geophone':
            ax[0, 0].set_xlabel('Particle Velocity (mm/s)')
        else:
            ax[0, 0].set_xlabel('Pressure (Pa)')
        ax[0, 0].set_ylabel('Count')

        rms_val = np.empty((all_traces.shape[0],))
        for nt in np.arange(all_traces.shape[0]):
            rms_val[nt] = rms(all_traces[nt,:])

        ax[0, 1].hist(rms_val, bins=30, log=True)
        if self.type_sensor.currentText() == 'Geophone':
            ax[0, 1].set_xlabel('RMS Trace Part. Vel. (mm/s)')
        else:
            ax[0, 1].set_xlabel('RMS Trace Pressure (Pa)')
        ax[0, 1].set_ylabel('Count')
        
        f, spectra = get_spectrum(all_traces, all_sample_rates)

        ax[1, 0].plot(f, spectra.mean(axis=0))
        ax[1, 0].set_xscale('log')
        ax[1, 0].set_yscale('log')
        if self.type_sensor.currentText() == 'Geophone':
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

        fig.suptitle(self.site_no.currentText()+', Train: '+self.train_list.currentText())

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.show()

    def analyze_timedeltas(self):
        mask = self.passage_times['Train'].str.contains('_').values
        mask[-1] = False
        times = (
            self.passage_times.loc[mask, 'passage_end'].values
            - self.passage_times.loc[mask, 'passage_start'].values
        )
        numeric_deltas = [time.total_seconds() for time in times]
        print(f"Min: {min(times)}")
        print(f"Max: {max(times)}")
        print(f"Mean: {np.mean(times)}")
        print(f"Std: {datetime.timedelta(seconds=np.std(numeric_deltas))}")
        plt.hist([d / 60 for d in numeric_deltas], bins=30)
        plt.xlabel("Durée du passage [min]")
        plt.ylabel("Count")
        plt.show()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = PyNDVP()

    sys.exit(app.exec_())
