# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:48:39 2024

@author: vatsa
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn
from scipy.fft import fftshift, fft

def rrcosfilter(num_taps, alpha, Ts, Fs):
    """
    Generates a Root Raised Cosine (RRC) filter.
    
    :param num_taps: Number of taps in the filter.
    :param alpha: Roll-off factor.
    :param Ts: Symbol period.
    :param Fs: Sampling frequency.
    :return: Filter coefficients and corresponding time vector.
    """
    T_delta = 1 / Fs
    t = np.arange(-num_taps // 2, num_taps // 2 + 1) * T_delta
    h = np.zeros_like(t)

    for i in range(len(t)):
        if t[i] == 0.0:
            h[i] = 1.0 + alpha * (4 / np.pi - 1)
        elif abs(t[i]) == Ts / (4 * alpha):
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            h[i] = (np.sin(np.pi * t[i] * (1 - alpha) / Ts) +
                    4 * alpha * t[i] / Ts * np.cos(np.pi * t[i] * (1 + alpha) / Ts)) / \
                   (np.pi * t[i] * (1 - (4 * alpha * t[i] / Ts) ** 2) / Ts)
    
    h /= np.sqrt(Ts)  # Normalize the filter
    return t, h

def generate_bpsk_signal(num_bits, Fs, Rs, alpha):
    """
    Generates a BPSK signal using a Root Raised Cosine filter.
    
    :param num_bits: Number of bits.
    :param Fs: Sampling frequency.
    :param Rs: Symbol rate.
    :param alpha: Roll-off factor.
    :return: Shaped BPSK signal, sampling frequency, and symbol rate.
    """
    bits = np.random.randint(0, 2, num_bits)
    symbols = 2 * bits - 1
    Ts = 1 / Rs
    num_taps = 101
    t, rrc_filter = rrcosfilter(num_taps, alpha, Ts, Fs)
    
    # Upsample symbols
    upsampled_symbols = upfirdn([1], symbols, up=int(Fs/Rs))
    
    # Apply RRC filter twice
    shaped_signal = upfirdn(rrc_filter, upsampled_symbols)
    shaped_signal = upfirdn(rrc_filter, shaped_signal)  # Apply the RRC filter again
    shaped_signal = shaped_signal[:len(upsampled_symbols)]  # Adjust length
    
    return shaped_signal, Fs, Rs

def plot_spectrum(signal, Fs):
    """
    Plots the power spectral density of the signal.
    
    :param signal: Input signal.
    :param Fs: Sampling frequency.
    """
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
    """
    Plots the eye diagram of the signal.
    
    :param signal: Input signal.
    :param samples_per_symbol: Number of samples per symbol.
    :param num_symbols: Number of symbols to display in the eye diagram.
    """
    span = samples_per_symbol * num_symbols
    offset = span // 3
    
    # Exclude the first and last portions of the signal to avoid boundary effects
    start = int(0.05 * len(signal))
    end = len(signal) - int(0.05 * len(signal))
    
    eye_data = np.array([signal[i:i + span] for i in range(start +3 , end - span +1 , samples_per_symbol)])
    
    plt.figure(figsize=(10, 6))
    plt.plot(eye_data.T, 'b', alpha=0.7)
    plt.title('Eye Diagram of BPSK with Double RRC Filter')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

# Parameters
num_bits = 100
Fs = 20e6  # Sampling frequency (samples per second)
Rs = 1e6  # Symbol rate (symbols per second)
alpha = 0.35  # Roll-off factor

# Generate BPSK signal
signal, Fs, Rs = generate_bpsk_signal(num_bits, Fs, Rs, alpha)

# Plot spectrum
plot_spectrum(signal, Fs)

# Plot eye diagram
plot_eye_diagram(signal, samples_per_symbol=int(Fs/Rs), num_symbols=2)
