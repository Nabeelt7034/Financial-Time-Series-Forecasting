# Pattern Recognition for Financial Time Series Forecasting

## 👤 Student Details

* **Name:** NABEEL T
* **Register Number:** LTCR24CS075

---

## 📌 Overview

This project analyzes financial time series data using signal processing and deep learning.

Stock prices of three companies are transformed into spectrograms using STFT, and a CNN model is used to predict future values.

---

## 📊 Data Used

* RELIANCE.NS
* TCS.NS
* INFY.NS

(Data collected using Yahoo Finance)

---

## 📁 Output

All outputs are saved in the `output/` folder:

* `time_series.png` → Time series plot
* `frequency_spectrum.png` → Frequency spectrum
* `spectrogram.png` → Time-frequency representation
* `cnn_architecture.png` → CNN model diagram
* `prediction.png` → Actual vs predicted graph
* `mse.txt` → Mean Squared Error value

---

## ▶️ How to Run

### Step 1: Clone the repository

```bash
git clone https://github.com/Nabeelt7034/Financial-Time-Series-Forecasting.git
cd Financial-Time-Series-Forecasting
```

### Step 2: Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirement.txt
```

### Step 4: Run the program

```bash
python src/predict.py
```

---
