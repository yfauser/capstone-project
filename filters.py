#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pywt
from scipy import signal
from scipy.signal import butter


def _maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def low_pass_filter(data, high_cutoff=10**4):
    """
    low pass filter for sinwave extraction
    """
    
    # 800,000 data points taken over 20 ms
    # Grid operates at 50hz, 0.02 * 50 = 1, so 800k samples in 20 milliseconds will capture one complete cycle
    n_samples = 800000
    # Sample duration is 20 miliseconds
    sample_duration = 0.02
    # Sample rate is the number of samples in one second, Sample rate will be 40mhz
    sample_rate = n_samples * (1 / sample_duration)
    
    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    sos = butter(10, high_cutoff, fs=sample_rate, btype='lowpass', output='sos')
    filtered_sig = signal.sosfilt(sos, data)

    return pd.Series(filtered_sig)



def high_pass_filter(data, low_cutoff=10**4):
    """
    From jackv https://www.kaggle.com/jackvial/dwt-signal-denoising
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
    This functin removes freqencies bellow 'low_cutoff', the default is 
    """
    
    # 800,000 data points taken over 20 ms
    # Grid operates at 50hz, 0.02 * 50 = 1, so 800k samples in 20 milliseconds will capture one complete cycle
    n_samples = 800000
    # Sample duration is 20 miliseconds
    sample_duration = 0.02
    # Sample rate is the number of samples in one second, Sample rate will be 40mhz
    sample_rate = n_samples * (1 / sample_duration)
    
    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    sos = butter(10, low_cutoff, fs=sample_rate, btype='highpass', output='sos')
    filtered_sig = signal.sosfilt(sos, data)

    return pd.Series(filtered_sig)


def denoise_signal(data, wavelet='db4', level=1):
    """
    From jackv https://www.kaggle.com/jackvial/dwt-signal-denoising

    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """
    
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec(data, wavelet, mode="per")
    
    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1/0.6745) * _maddest(coeff[-level])

    # Calculte the univeral threshold
    uthresh = sigma * np.sqrt(2*np.log(len(data)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    
    # Reconstruct the signal using the thresholded coefficients
    denoised_sig = pywt.waverec(coeff, wavelet, mode='per')

    return pd.Series(denoised_sig)


def _find_xcut(data, min_idx, backwards=False):
    index = None
    if backwards:
        for index, value in data[min_idx::-1].items():
            if value >= 0 and data[index+1] < 0:
                return index
    else:
        for index, value in data[min_idx:].items():
            if value >= 0 and data[index-1] < 0:
                return index


def sinewave_shift(data, start_point=3000):
    """
    This function finds the lowest amplitute of the 50Hz sinwave,
    cuts the sinwave at that point and shifts all the data starting at that point 
    from left to right. This creates a sinwave that is mostly time synced
    and therefore all samples can be cut at the same point to reduce input data sizes
    """
    clean_sinewave = low_pass_filter(data)
    wave_min = clean_sinewave.idxmin()
    if _find_xcut(clean_sinewave, wave_min):
        cutpoint = _find_xcut(clean_sinewave, wave_min)
    else:
        cutpoint = _find_xcut(clean_sinewave, wave_min, backwards=True)
    synced_sample = pd.concat([data[cutpoint:], data[:cutpoint]])
    synced_sample.reset_index(drop=True, inplace=True)
    
    return synced_sample, cutpoint


def drop_data(data, drop_ranges=None):
    """
    This function takes a series as input and drops the ranges passed
    as a list of tupples with start and stop index of data to be dropped.
    It returns the reindexed resulting series after dropping the desired parts
    """
    if not drop_ranges:
        drop_ranges=[(0, 50000), (300000, 450000), (700000, 800000)]

    data_to_drop = []
    for drop_range in drop_ranges:
        data_to_drop += list(range(drop_range[0],drop_range[1]))

    return data.drop(labels=data_to_drop).reset_index(drop=True)


def count_peaks(data, threshold=1):
    """
    Counts the number of absolute values where a threshold is exceeded  
    """
    peaks = data[data.abs() > threshold]
    return len(peaks)

