

import numpy as np
import matplotlib.pyplot as plt

# Parameters
modulation = 'QPSK'  # Choose 'BPSK' or 'QPSK'
num_bits = 1000000   # Increase number of bits for accuracy
snr_range = np.arange(0, 20, 2)  # SNR range in dB

# Generate random binary data
data = np.random.randint(0, 2, num_bits)

# Modulate data
if modulation == 'BPSK':
    modulated_signal = 2 * data - 1  # BPSK: Map 0 -> -1, 1 -> +1
elif modulation == 'QPSK':
    data = data.reshape(-1, 2)       # Group bits into pairs
    modulated_signal = np.array([complex(2*b[0]-1, 2*b[1]-1) for b in data])  # QPSK mapping

# BER calculation
ber_results = []
for snr_db in snr_range:
    snr_linear = 10**(snr_db / 10)
    noise_power = np.mean(np.abs(modulated_signal)**2) / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*modulated_signal.shape) + 
                                        1j*np.random.randn(*modulated_signal.shape))
    received_signal = modulated_signal + noise
    
    # Demodulate received signal
    if modulation == 'BPSK':
        demodulated_data = (received_signal.real > 0).astype(int)
    elif modulation == 'QPSK':
        demodulated_data = np.array([[int(r.real > 0), int(r.imag > 0)] for r in received_signal]).flatten()

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

