import numpy as np
import matplotlib.pyplot as plt

# Data arrays
freq_offset = np.array([0, 10, 20, 30, 50, 70, 100, 120, 150, 200, 280, 350, 400, 500,
                        700, 1000, 1500, 2000, 3000, 5000, 10000, 15000, 30000, 50000,
                        100000, 200000, 300000, 500000, 1000000])
EVM = np.array([0.65, 0.66, 0.72, 0.73, 0.76, 0.78, 0.93, 0.999, 1.106, 1.351, 1.885,
                2.247, 2.528, 3.107, 4.276, 6.018, 8.751, 12.013, 16.915, 24.852,
                28.465, 28.529, 27.970, 29.231, 28.957, 29.027, 40.673, 23.488, 53.913])

# Filter out zero to enable logarithmic scale
mask = freq_offset > 0
freq = freq_offset[mask]
evm = EVM[mask]

# Plot with logarithmic x-axis
plt.figure(figsize=(10, 6))
plt.semilogx(freq, evm, marker='o', linestyle='-')
plt.title('Impact of Frequency Offset on EVM')
plt.xlabel('Frequency Offset (Hz, log scale)')
plt.ylabel('EVM (dB)')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
