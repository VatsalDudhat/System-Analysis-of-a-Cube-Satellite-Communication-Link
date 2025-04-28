# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 20:19:00 2025

@author: vatsa
"""

import pandas as pd
import matplotlib.pyplot as plt

# Data extracted from the images (C/N ratio, EVM, and SNR)
data = {
    "Eb/No (dB)": [38.51, 33.51, 28.51, 23.51, 18.51, 13.51, 8.51, 3.51, -1.49],
    "EVM (%)": [0.627, 1.606, 2.716, 4.67, 8.314, 14.98, 27.57, 44.95, 56.8],
    "SNR (dB)": [44.05, 35.41, 31.23, 26.62, 21.60, 16.49, 11.27, 6.94, 5.02],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plotting EVM vs Eb/No
axs[0].plot(df["Eb/No (dB)"], df["EVM (%)"], marker='o')
axs[0].set_title("EVM vs Energy per bit over noise density")
axs[0].set_xlabel("Eb/No (dB)")
axs[0].set_ylabel("EVM (%)")
axs[0].grid(True)

# Plotting SNR vs Eb/No
axs[1].plot(df["Eb/No (dB)"], df["SNR (dB)"], marker='s', color='orange')
axs[1].set_title("SNR vs Energy per bit over noise density")
axs[1].set_xlabel("Eb/No (dB)")
axs[1].set_ylabel("SNR (dB)")
axs[1].grid(True)

plt.tight_layout()
plt.show()
