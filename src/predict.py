# =========================================================
# PATTERN RECOGNITION FOR FINANCIAL TIME SERIES FORECASTING
# FINAL IMPROVED VERSION (ASSIGNMENT PERFECT)
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model

# =========================================================
# CREATE OUTPUT FOLDER
# =========================================================
os.makedirs("output", exist_ok=True)

# =========================================================
# TASK 1: DATA PREPARATION
# =========================================================
symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

data = []
for sym in symbols:
    df_temp = yf.download(sym, start="2019-01-01", end="2024-01-01")
    df_temp = df_temp[['Close']]
    df_temp.columns = [sym]
    data.append(df_temp)

df = pd.concat(data, axis=1).dropna()

# Normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# =========================================================
# FIGURE 1: TIME SERIES
# =========================================================
plt.figure(figsize=(10,5))
for i in range(scaled.shape[1]):
    plt.plot(df.index, scaled[:, i], label=df.columns[i])
plt.legend()
plt.xlabel("Time")
plt.ylabel("Normalized Value")
plt.title("Time Series Data")
plt.savefig("output/time_series.png", dpi=300)
plt.close()

# =========================================================
# TASK 2: FOURIER TRANSFORM
# =========================================================
signal = scaled[:, 0]

N = len(signal)
fft_vals = np.abs(fft(signal))
freqs = fftfreq(N)

# =========================================================
# FIGURE 2: FREQUENCY SPECTRUM
# =========================================================
plt.figure()
plt.plot(freqs[:N//2], fft_vals[:N//2])
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("Frequency Spectrum")
plt.savefig("output/frequency_spectrum.png", dpi=300)
plt.close()

# =========================================================
# TASK 2: STFT (SPECTROGRAM)
# =========================================================
f, t, Zxx = stft(signal, nperseg=64)
spectrogram = np.abs(Zxx)**2

# =========================================================
# FIGURE 3: SPECTROGRAM
# =========================================================
plt.figure()
plt.pcolormesh(t, f, spectrogram)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("Spectrogram")
plt.savefig("output/spectrogram.png", dpi=300)
plt.close()

# =========================================================
# TASK 3: DATASET (SLIDING WINDOW)
# =========================================================
X = []
y = []
window = 64

for i in range(len(signal) - window - 1):
    segment = signal[i:i+window]

    f, t, Zxx = stft(segment, nperseg=64)
    Sxx = np.abs(Zxx)**2

    X.append(Sxx)
    y.append(signal[i+window])

X = np.array(X)
y = np.array(y)

# Add channel dimension
X = X[..., np.newaxis]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================================================
# TASK 3: CNN MODEL (IMPROVED BUT SAFE)
# =========================================================
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', padding='same',
                  input_shape=X.shape[1:]),

    layers.MaxPooling2D((2,2)),

    # small improvement: more feature learning but safe
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),

    # important: helps reduce over-smooth prediction
    layers.Dense(32, activation='relu'),

    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# =========================================================
# FIGURE 4: CNN ARCHITECTURE
# =========================================================
plot_model(model,
           to_file="output/cnn_architecture.png",
           show_shapes=True)

# =========================================================
# TRAIN MODEL (slightly more epochs for better fit)
# =========================================================
model.fit(X_train, y_train, epochs=15, verbose=1)

# =========================================================
# TASK 3: PREDICTION
# =========================================================
pred = model.predict(X_test)

# =========================================================
# TASK 4: ANALYSIS
# =========================================================

# =========================================================
# FIGURE 5: ACTUAL vs PREDICTED
# =========================================================
plt.figure()
plt.plot(y_test, label="Actual")
plt.plot(pred, label="Predicted")
plt.legend()
plt.title("Prediction vs Actual")
plt.savefig("output/prediction.png", dpi=300)
plt.close()

# =========================================================
# MSE
# =========================================================
mse = np.mean((pred - y_test)**2)
print("MSE:", mse)

with open("output/mse.txt", "w") as f:
    f.write(f"MSE: {mse}")

print("✅ SUCCESS: All outputs saved in 'output/' folder")