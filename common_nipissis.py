#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import socket
if socket.gethostname() == 'harricana':
    root_dir = '/Volumes/Data4/nipissis/'
else:
    root_dir = '/Users/giroux/Desktop/nipissis/'


# Hydrophones
#
# -160 dB ± 3 dB re 1V/μPa
#
sensitivity_h = np.array([10**(-160.0/20.0) for x in range(24)])  # V/μPa
sensitivity_h *= 1e6    # V/Pa

# Geophones
#
sensitivity_g = [23.4 for x in range(24)]  # V/m/s


#
#  Some functions
#
def rms(x):
    return np.sqrt(np.sum(x*x) / x.size)
