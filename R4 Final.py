# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 11:26:53 2024

@author: vatsa
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn, firwin
from scipy.fft import fftshift, fft

def rrcosfilter(num_taps, alpha, Ts, Fs):
    T_delta = 1 / Fs
    t = np.arange(-num_taps // 2, num_taps // 2 + 1) * T_delta
    h = np.zeros_like(t)

    for i in range(len(t)):
        if t[i] == 0.0:
            h[i] = 1.0 + alpha * (4 / np.pi - 1)
        elif abs(t[i]) == Ts / (4 * alpha):
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * (np.sin(np.pi / (4 * alpha))) + 
                (1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))
            )
        else:
            h[i] = (np.sin(np.pi * t[i] * (1 - alpha) / Ts) +
                    4 * alpha * t[i] / Ts * np.cos(np.pi * t[i] * (1 + alpha) / Ts)) / \
                   (np.pi * t[i] * (1 - (4 * alpha * t[i] / Ts) ** 2) / Ts)
    
    h /= np.sqrt(Ts)
    return t, h

def generate_bpsk_signal(num_bits, Fs, Rs, alpha):
    bits = np.random.randint(0, 2, num_bits)
    symbols = 2 * bits - 1
    Ts = 1 / Rs
    num_taps = 101
    t, rrc_filter = rrcosfilter(num_taps, alpha, Ts, Fs)
    upsampled_symbols = upfirdn([1], symbols, up=int(Fs/Rs))
    shaped_signal = upfirdn(rrc_filter, upsampled_symbols)
    shaped_signal = shaped_signal[:len(upsampled_symbols)]  # Adjust length
    return shaped_signal, Fs, Rs

def plot_spectrum(signal, Fs):
    f = np.linspace(-Fs / 2, Fs / 2, len(signal))
    S = fftshift(fft(signal))
    plt.figure(figsize=(10, 6))
    plt.plot(f / 1e6, 20 * np.log10(np.abs(S) / max(np.abs(S))))
    plt.title('Power Spectral Density of BPSK with RRC Filter')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power/Frequency (dB)')
    plt.grid(True)
    plt.show()

def plot_eye_diagram(signal, samples_per_symbol, num_symbols):
    span = samples_per_symbol * num_symbols
    offset = span // 2
    
    # Leave space at the start and end by excluding the first and last 0.5% of the signal
    start = int(0.005 * len(signal))
    end = len(signal) - int(0.005 * len(signal))

    eye_data = np.array([signal[i:i + span] for i in range(start, end - span + 1, samples_per_symbol)])
    
    plt.figure(figsize=(10, 6))
    plt.plot(eye_data.T, 'b')
    plt.title('Eye Diagram of BPSK with RRC Filter')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

# Parameters
num_bits = 10000
Fs = 20e6  # Sampling rate (samples per second)
Rs = 1e6  # Symbol rate (symbols per second)
alpha = 0.35  # Roll-off factor

# Generate BPSK signal
signal, Fs, Rs = generate_bpsk_signal(num_bits, Fs, Rs, alpha)

# Plot spectrum
plot_spectrum(signal, Fs)

# Plot eye diagram
plot_eye_diagram(signal, samples_per_symbol=int(Fs/Rs), num_symbols=2)

