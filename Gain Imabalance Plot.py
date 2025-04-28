import matplotlib.pyplot as plt

# Re-defining the data
gain_imbalance = [0.000, 0.100, 0.200, 0.300, 0.400, 0.500, 0.700, 1.000]
snr = [44.05, 31.46, 30.99, 30.08, 29.26, 28.29, 26.41, 23.99]
evm = [0.627, 2.620, 2.822, 3.122, 3.454, 3.851, 4.781, 6.320]

# Create subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot Gain Imbalance vs SNR
axs[0].plot(gain_imbalance, snr, marker='o')
axs[0].set_title("Gain Imbalance vs SNR")
axs[0].set_xlabel("Gain Imbalance (dB)")
axs[0].set_ylabel("SNR (dB)")
axs[0].grid(True)

# Plot Gain Imbalance vs EVM
axs[1].plot(gain_imbalance, evm, marker='o', color='orange')
axs[1].set_title("Gain Imbalance vs EVM")
axs[1].set_xlabel("Gain Imbalance (dB)")
axs[1].set_ylabel("EVM (%)")
axs[1].grid(True)

plt.tight_layout()
plt.show()
