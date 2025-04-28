# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 22:11:45 2024

@author: vatsal
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn  # Import upfirdn from scipy.signal
from scipy.fft import fftshift, fft

# Load the numpy arrays from the .npy files
x = np.load('x.npy')
y = np.load('y.npy')

# Check the shape of the arraysprint(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")

# Flatten the arrays if they have extra dimensions
x = x.flatten()
y = y.flatten()

# Print some sample values and ranges
print(f"x first 5 values: {x[:5]}")
print(f"y first 5 values: {y[:5]}")
print(f"x last 5 values: {x[-5:]}")
print(f"y last 5 values: {y[-5:]}")
print(f"x min: {x.min()}, x max: {x.max()}")
print(f"y min: {y.min()}, y max: {y.max()}")

# Plot the data points
plt.figure(figsize=(10, 6))
plt.plot(x, y, '*', label='AWR Plot')  # 'o' for markers
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y vs x with BPSK Power Spectrum Overlay')
plt.legend()
plt.grid(True)

# Set limits based on data ranges
plt.xlim(x.min(), x.max())
plt.ylim(y.min(), y.max())

# ---- Power Spectrum Overlay with Averaging ----

def generate_bpsk_signal(num_bits, Fs, Rs, alpha):
    """
    Generates a BPSK signal using a raised cosine filter.
    """
    bits = np.random.randint(0, 2, num_bits)
    symbols = 2 * bits - 1
    Ts = 1 / Rs
    num_taps = 101
    t, rrc_filter = rrcosfilter(num_taps, alpha, Ts, Fs)
    upsampled_symbols = upfirdn([1], symbols, up=int(Fs / Rs))
    shaped_signal = upfirdn(rrc_filter, upsampled_symbols)
    shaped_signal = shaped_signal[:len(upsampled_symbols)]  # Adjust length
    return shaped_signal, Fs, Rs

def rrcosfilter(num_taps, alpha, Ts, Fs):
    """
    Generates a raised cosine (RRC) filter.
    """
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

def plot_averaged_power_spectrum(signal, Fs, x_min, x_max, y_min, y_max, num_segments=10):
    """
    Plots the averaged power spectral density (PSD) of the BPSK signal, scaled and
    overlaid on the x-y plot.
    
    The power spectrum is averaged over 'num_segments' and scaled to fit within the
    limits of the x-y plot.
    """
    segment_length = len(signal) // num_segments
    f = np.linspace(-Fs / 2, Fs / 2, segment_length)
    avg_psd = np.zeros_like(f)

    for i in range(num_segments):
        segment = signal[i * segment_length: (i + 1) * segment_length]
        S = fftshift(fft(segment))
        avg_psd += np.abs(S) ** 2

    avg_psd /= num_segments

    # Normalize the power spectrum to match the y-range of the other plot
    avg_psd = 20 * np.log10(avg_psd / np.max(avg_psd))
    
    # Scale and shift the power spectrum to fit in the (x_min, x_max) and (y_min, y_max) ranges
    scaled_frequencies = np.interp(f, (f.min(), f.max()), (x_min, x_max))  # Map frequencies to x-axis range
    scaled_power = np.interp(avg_psd, (avg_psd.min(), avg_psd.max()), (y_min, y_max))  # Map power to y-axis range

    # Plot the scaled averaged power spectrum on the same plot
    plt.plot(scaled_frequencies, scaled_power, 'r-', label='python plot')
    plt.ylabel('Power (dB)')
    plt.legend(loc='upper right')

# Parameters for BPSK signal generation
num_bits = 10000
Fs = 20e6  # Sampling rate (samples per second)
Rs = 1e6   # Symbol rate (symbols per second)
alpha = 0.35  # Roll-off factor

# Generate BPSK signal
bpsk_signal, Fs, Rs = generate_bpsk_signal(num_bits, Fs, Rs, alpha)

# Overlay the averaged power spectrum on the same plot
plot_averaged_power_spectrum(bpsk_signal, Fs, x.min(), x.max(), y.min(), y.max(), num_segments=10)

plt.show()

