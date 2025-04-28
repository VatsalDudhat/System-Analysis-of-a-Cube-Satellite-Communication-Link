# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 13:41:47 2025

@author: vatsa
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
modulation = 'QPSK'  # Modulation type
num_bits = 1000000    # Number of bits (ensure it's divisible by 2 for QPSK)
snr_range = np.arange(0, 20, 2)  # SNR range in dB

# Generate random binary data
data = np.random.randint(0, 2, num_bits)

# Reshape data for QPSK (group into pairs)
data_pairs = data.reshape(-1, 2)  # Each pair represents one QPSK symbol

# Map binary pairs to QPSK symbols (complex numbers)
modulated_signal = np.array([complex(2*b[0]-1, 2*b[1]-1) for b in data_pairs])

# BER calculation
ber_results = []
for snr_db in snr_range:
    snr_linear = 10**(snr_db / 10)  # Convert SNR from dB to linear scale
    noise_power = np.mean(np.abs(modulated_signal)**2) / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*modulated_signal.shape) + 
                                        1j*np.random.randn(*modulated_signal.shape))  # AWGN noise
    
    # Received signal with noise
    received_signal = modulated_signal + noise
    
    # Demodulate received signal (decision boundaries for QPSK)
    demodulated_data_pairs = np.array([[int(r.real > 0), int(r.imag > 0)] for r in received_signal])
    
    # Flatten demodulated pairs back into a single binary stream
    demodulated_data = demodulated_data_pairs.flatten()

    # Calculate BER
    bit_errors = np.sum(data != demodulated_data)
    ber = bit_errors / num_bits
    ber_results.append(ber)

# Plot BER vs SNR
plt.figure()
plt.semilogy(snr_range, ber_results, marker='o')
plt.title(f'BER vs SNR ({modulation})')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate')
plt.grid(True, which='both', linestyle='--')
plt.show()
