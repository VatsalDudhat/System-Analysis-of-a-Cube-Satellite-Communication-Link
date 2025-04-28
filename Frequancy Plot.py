import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn
from scipy.fft import fftshift, fft

def rrcosfilter(num_taps, alpha, Ts, Fs):
    T_delta = 1/Fs
    t = np.arange(-num_taps//2, num_taps//2 + 1) * T_delta
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 + alpha*(4/np.pi - 1)
        elif np.isclose(abs(ti), Ts/(4*alpha)):
            h[i] = (alpha/np.sqrt(2)) * (
                (1 + 2/np.pi)*np.sin(np.pi/(4*alpha)) +
                (1 - 2/np.pi)*np.cos(np.pi/(4*alpha))
            )
        else:
            num = (np.sin(np.pi*ti*(1-alpha)/Ts) +
                   4*alpha*ti/Ts * np.cos(np.pi*ti*(1+alpha)/Ts))
            den = np.pi*ti*(1 - (4*alpha*ti/Ts)**2)/Ts
            h[i] = num/den
    return t, h/np.sqrt(Ts)

def generate_bpsk_signal(num_bits, Fs, Rs, alpha):
    bits    = np.random.randint(0,2,num_bits)
    symbols = 2*bits - 1
    Ts      = 1/Rs
    _, rrc  = rrcosfilter(101, alpha, Ts, Fs)
    up      = int(Fs/Rs)
    upsym   = upfirdn([1], symbols, up=up)
    shaped  = upfirdn(rrc, upsym)
    return shaped[:len(upsym)]

def raw_averaged_psd(signal, Fs, num_segments=20):
    L = len(signal)//num_segments
    f = np.linspace(-Fs/2, Fs/2, L)
    psd = np.zeros(L)
    for i in range(num_segments):
        seg = signal[i*L:(i+1)*L]
        psd += np.abs(fftshift(fft(seg)))**2
    return f, psd/num_segments

def smooth(data, window_len=15):
    return np.convolve(data, np.ones(window_len)/window_len, mode='same')

# --- Parameters ---
Fs = 20e6; Rs = 1e6; alpha = 0.35; num_bits = 10000
center_freq_mhz = 2385

# Generate base waveform once
base_sig = generate_bpsk_signal(num_bits, Fs, Rs, alpha)

# Power levels and target peak alignments
power_levels = [13.1, 13.0, 11.5]
target_peaks = [13.1, 13.0, 11.5]  # two at 13, one at 12
labels       = ['Python', 'AWR', 'Hardware']
colors       = ['tab:blue', 'tab:orange', 'tab:green']
linestyles   = ['-', '-', ':']
markers      = ['*', None, 's']  # square marker for hardware
markevery    = [5, 50, 50]    # hardware markers every 50 points

plt.figure(figsize=(10, 6))
window_len = 15
half = window_len // 2

for P, T, lbl, c, ls, m, me in zip(power_levels, target_peaks, labels, colors, linestyles, markers, markevery):
    # scale waveform by nominal level
    scale = 10**((P - 13.0)/20)
    sig = base_sig * scale
    # compute PSD and align peak
    f, psd_lin = raw_averaged_psd(sig, Fs, num_segments=20)
    psd_db = 10 * np.log10(psd_lin/psd_lin.max()) + T
    # smooth and trim edges
    psd_s = smooth(psd_db, window_len=window_len)
    f_trim = f[half:-half]
    psd_trim = psd_s[half:-half]
    # plot with distinct style
    plt.plot(
        f_trim/1e6 + center_freq_mhz,
        psd_trim,
        label=lbl,
        color=c,
        linestyle=ls,
        marker=m,
        markevery=me,
        linewidth=2
    )

plt.title('QPSK Spectra @ 2.385 GHz')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (dBm)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
