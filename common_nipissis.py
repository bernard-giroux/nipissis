#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyfftw

import socket

if socket.gethostname() == 'harricana':
    root_dir = '/Volumes/Data4/nipissis/'
elif 'saintemarguerite' in socket.gethostname():
    root_dir = '/Volumes/Nipissis/'
elif socket.gethostname() == 'LAPTOP-K01UUA9T':
    root_dir = 'E:/nipissis/'
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

def integrate(x, fs):
    Nr = x.size
    Nc = int(Nr/2+1)
    if x.dtype is np.dtype('float64'):
        a = pyfftw.empty_aligned(Nr, dtype='float64')
        b = pyfftw.empty_aligned(Nc, dtype='complex128')
        c = pyfftw.empty_aligned(Nr, dtype='float64')
    elif  x.dtype is np.dtype('float32'):
        a = pyfftw.empty_aligned(Nr, dtype='float32')
        b = pyfftw.empty_aligned(Nc, dtype='complex64')
        c = pyfftw.empty_aligned(Nr, dtype='float32')
    else:
        raise TypeError()
    
    for n in np.arange(x.size):
        a[n] = x[n]
    
    domega = 2*np.pi*fs / (Nr*Nr)
    omega = domega * np.arange(Nc)
    
    fft_object = pyfftw.FFTW(a, b)
    ifft_object = pyfftw.FFTW(b, c, direction='FFTW_BACKWARD')

    fft_object()
    b[0] *= 0.0
    for n in np.arange(1, Nc):
        b[n] /= 1j*omega[n]
    
    ifft_object()
    y = np.empty(x.shape, x.dtype)
    for n in np.arange(c.size):
        y[n] = c[n]/Nr
    
    return y
