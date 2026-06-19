import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny

# -----------------------------
# 1) generate FM-CW signal
# -----------------------------
fs = 2000
t = np.linspace(0, 2, 2*fs)

f0 = 50
k = 80   # chirp rate

signal = np.cos(2*np.pi*(f0*t + 0.5*k*t**2))

# add noise
noise = 0.4*np.random.randn(len(t))
x = signal + noise

# -----------------------------
# 2) Time-Frequency representation
# -----------------------------
f, tt, Sxx = spectrogram(x, fs=fs, nperseg=128, noverlap=100)

Sxx_db = 10*np.log10(Sxx + 1e-10)

# normalize image
img = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min())

# -----------------------------
# 3) Edge detection on TF image
# -----------------------------
edges = canny(img, sigma=2)

# -----------------------------
# 4) Hough transform to detect line
# -----------------------------
h, theta, d = hough_line(edges)
accum, angles, dists = hough_line_peaks(h, theta, d)

# -----------------------------
# 5) Estimate IF line
# -----------------------------
lines = []
for angle, dist in zip(angles, dists):
    x_vals = np.arange(img.shape[1])
    y_vals = (dist - x_vals*np.cos(angle)) / np.sin(angle)
    lines.append((x_vals, y_vals))

# -----------------------------
# Plot results
# -----------------------------
plt.figure(figsize=(12,8))

plt.subplot(221)
plt.plot(t, x)
plt.title("FM-CW Signal")

plt.subplot(222)
plt.pcolormesh(tt, f, Sxx_db, shading='gouraud')
plt.title("Spectrogram")
plt.ylabel("Frequency")

plt.subplot(223)
plt.imshow(edges, cmap='gray', aspect='auto')
plt.title("Edges in Time-Frequency Image")

plt.subplot(224)
plt.imshow(img, cmap='jet', aspect='auto')
for x_vals, y_vals in lines:
    plt.plot(x_vals, y_vals, '-r')
plt.title("Estimated IF using Hough Transform")

plt.tight_layout()
plt.show()
