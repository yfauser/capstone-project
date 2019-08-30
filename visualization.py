#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.ticker as ptck
import numpy as np
from scipy import signal

def plot_phase_time(data, subset=None, ylimit=[-50, 50]):
    if not subset:
        subset = len(data)
    
    fig, ax1 = plt.subplots(figsize=(20, 10))
    samples = np.array(data[:subset].index + 1)
    time = samples / 40000  # sampling rate is 40MS/s (40Mhz), so total time in ms is 40Mhz / 1000 = 40khz 
    values = data[:subset]
    
    ax1.plot(time, values)
    ax1.set(xlabel='Time (ms)', ylabel='Signal Amplitute (voltage)', title='Measurement Sample')
    ax1.set_ylim(ylimit)
    ax1.grid()


def plot_phase_sample(data, subset=None, ylimit=[-50, 50]):
    if not subset:
        subset = len(data)
    
    fig, ax1 = plt.subplots(figsize=(20, 10))
    samples = np.array(data[:subset].index + 1)
    values = data[:subset]
    
    ax1.plot(samples, values)
    ax1.set(xlabel='Sample', ylabel='Signal Amplitute (voltage)', title='Measurement Sample')
    ax1.set_ylim(ylimit)
    ax1.grid()


def plot_specto(data, subset=None, vmin=-151):
    if not subset:
        subset = len(data)

    # 800,000 data points taken over 20 ms
    # Grid operates at 50hz, 0.02 * 50 = 1, so 800k samples in 20 milliseconds will capture one complete cycle
    n_samples = 800000
    # Sample duration is 20 miliseconds
    sample_duration = 0.02
    # Sample rate is the number of samples in one second, Sample rate will be 40mhz
    sample_rate = n_samples * (1 / sample_duration)

    fig, ax1 = plt.subplots(figsize=(20, 10))
    samples = np.array(data[:subset].index + 1)
    values = data[:subset]
    
    Pxx, freqs, bins, im = ax1.specgram(values, Fs=sample_rate, vmin=vmin)
    ax1.set(xlabel='Time (ms)', ylabel='frequency kHz', title='Measurement Spectogram')
    cbar = fig.colorbar(im)
    cbar.set_label('Intensity dB')

    scale = 1e3 # KHz
    yticks = ptck.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
    ax1.yaxis.set_major_formatter(yticks)

    xticks = ptck.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1000))
    ax1.xaxis.set_major_formatter(xticks)


def save_specto(data, filename, filepath=None, subset=None, vmin=-151):
    if not subset:
        subset = len(data)
    if not filepath:
        filepath = ''

    # 800,000 data points taken over 20 ms
    # Grid operates at 50hz, 0.02 * 50 = 1, so 800k samples in 20 milliseconds will capture one complete cycle
    n_samples = 800000
    # Sample duration is 20 miliseconds
    sample_duration = 0.02
    # Sample rate is the number of samples in one second, Sample rate will be 40mhz
    sample_rate = n_samples * (1 / sample_duration)

    fig, ax1 = plt.subplots(figsize=(22.4, 22.4))
    samples = np.array(data[:subset].index + 1)
    values = data[:subset]
    
    Pxx, freqs, bins, im = ax1.specgram(values, Fs=sample_rate, vmin=vmin)

    fig.patch.set_visible(False)
    ax1.axis('off')

    fig.savefig('{}/{}'.format(filepath, filename), dpi=10)
    plt.close(fig)


def plot_wavelet(data, subset=None):
    if not subset:
        subset = len(data)

    samples = np.array(data[:subset].index + 1)
    sig = data[:subset]
    scales = np.arange(1, 31)

    fig, ax1 = plt.subplots(figsize=(20, 10))

    cwtmatr = signal.cwt(sig, signal.morlet, scales)
    power = (abs(cwtmatr)) ** 2

    plt.imshow(np.log2(power), aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())

