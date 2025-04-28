# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 20:49:46 2024

@author: vatsa
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from scipy.fftpack import fft, ifft

# Constants
freq = 2e9  # Frequency in Hz
symbol_rate = 1e6  # Symbol rate in symbols/sec
sample_rate = 10e6  # Sampling rate in Hz
c = 3e8  # Speed of light in m/s
distance = 500e3  # Distance in meters (500 km)
velocity = 7500  # Relative velocity in m/s (Doppler shift simulation)

# Functions

def generate_signal(duration=1):
    """Generate a random signal."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.random.choice([1, -1], size=len(t))
    return t, signal

def add_noise(signal, snr_db):
    """Add AWGN noise to the signal."""
    snr_linear = 10**(snr_db / 10)
    power_signal = np.mean(signal**2)
    power_noise = power_signal / snr_linear
    noise = np.sqrt(power_noise) * np.random.randn(len(signal))
    return signal + noise

def doppler_effect(signal, t):
    """Simulate Doppler shift."""
    doppler_freq_shift = (velocity / c) * freq
    shifted_signal = signal * np.cos(2 * np.pi * doppler_freq_shift * t)
    return shifted_signal

def multipath_effect(signal, num_paths=3):
    """Simulate multipath effect."""
    delays = np.random.randint(1, 10, size=num_paths)
    multipath_signal = signal.copy()
    for delay in delays:
        multipath_signal += np.roll(signal, delay)
    return multipath_signal / (num_paths + 1)

def plot_spectrum(signal, title="Spectrum"):
    """Plot the spectrum of the signal."""
    fft_signal = fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/sample_rate)
    plt.figure()
    plt.plot(freqs[:len(freqs)//2], np.abs(fft_signal[:len(freqs)//2]))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

def bit_error_rate(original, received):
    """Calculate the bit error rate (BER)."""
    errors = np.sum(original != received)
    return errors / len(original)

# Main Script
duration = 0.01  # 10 ms for quick simulation
t, original_signal = generate_signal(duration)

# Add noise
noisy_signal = add_noise(original_signal, snr_db=10)

# Doppler effect
doppler_signal = doppler_effect(original_signal, t)

# Multipath effect
multipath_signal = multipath_effect(original_signal)

# Plot original and processed signals
plt.figure(figsize=(10, 6))
plt.plot(t[:1000], original_signal[:1000], label="Original Signal")
plt.plot(t[:1000], noisy_signal[:1000], label="Noisy Signal")
plt.plot(t[:1000], doppler_signal[:1000], label="Doppler Shift Signal")
plt.plot(t[:1000], multipath_signal[:1000], label="Multipath Signal")
plt.title("Signal Analysis")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Spectrum Analysis
plot_spectrum(original_signal, "Original Signal Spectrum")
plot_spectrum(noisy_signal, "Noisy Signal Spectrum")
plot_spectrum(doppler_signal, "Doppler Effect Spectrum")
plot_spectrum(multipath_signal, "Multipath Effect Spectrum")

# BER Calculation
received_signal = np.sign(noisy_signal)
ber = bit_error_rate(original_signal, received_signal)
print(f"Bit Error Rate (BER): {ber:.4f}")
