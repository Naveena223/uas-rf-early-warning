# UAS Passive RF Early-Warning System

A real-time passive RF monitoring pipeline that detects Unmanned Aerial Systems (UAS/drones)
by analysing radio frequency signatures using ensemble Machine Learning and DSP techniques.

Built with Python · scikit-learn · Dash · NumPy · SciPy

---

## What This System Does

- Passively monitors RF spectrum (no active transmission)
- Extracts 28 signal features across spectral, temporal, modulation, and cyclostationary categories
- Classifies signals as **UAS-LIKE**, **UNKNOWN**, or **NON-UAS** using Random Forest + Gradient Boosting
- Maintains a signature library of detected UAS RF fingerprints
- Displays real-time results on a live web dashboard with auto-refresh

---

## Project Structure
uas-rf-early-warning/
├── pipeline.py          # Main orchestrator — runs the full detection chain
├── parsers.py           # RF data parsers (SigMF, NPZ, CSV, image spectrograms)
├── feature_extractor.py # 28-feature RF signal extractor
├── classifier.py        # Ensemble ML classifier (RF + GBT + rule-based)
├── alert_engine.py      # Alert generation, deduplication, latency tracking
├── library.py           # UAS signature library (persistent JSONL store)
├── trainer.py           # Model training script
├── dataset_generator.py # Synthetic RF dataset generator
├── dashboard.py         # Live Dash web dashboard
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker container definition
└── README.md

---

## Quick Start (Recommended)

### Option A — Run with Python directly

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/uas-rf-early-warning.git
cd uas-rf-early-warning
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Generate test RF input data**

Create a file called `generate_test_input.py` and run it:
```python
import numpy as np, os
os.makedirs("test_data", exist_ok=True)
spec = np.random.randn(50, 512).astype(np.float32) * 0.1
spec[20:25, 100:120] = 5.0
spec[20:25, 250:270] = 5.0
spec[20:25, 400:420] = 5.0
freq_axis = np.linspace(2400e6, 2500e6, 50)
time_axis = np.linspace(0, 1.0, 512)
metadata  = {'center_freq_hz': 2440e6, 'sample_rate_hz': 100e6}
np.savez("test_data/uas_test.npz",
         spectrogram=spec, freq_axis=freq_axis,
         time_axis=time_axis, metadata=metadata)
print("Done!")
```
```bash
python generate_test_input.py
```

**4. Run the detection pipeline**
```bash
python run_pipeline.py
```

**5. Launch the dashboard**
```bash
python dashboard.py --log-dir logs/ --library-path library/uas_signatures.jsonl
```

**6. Open your browser**
http://localhost:8050/

---

### Option B — Run with Docker

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/uas-rf-early-warning.git
cd uas-rf-early-warning
```

**2. Build the Docker image**
```bash
docker build -t uas-rf-dashboard .
```

**3. Run the container**
```bash
docker run -p 8050:8050 uas-rf-dashboard
```

**4. Open your browser**
http://localhost:8050/

---

## Dashboard Features

| Panel | Description |
|---|---|
| KPI Cards | UAS alert count, Unknown count, Mean confidence, Max latency, Library size |
| Alert Timeline | Confidence vs. time scatter plot with threshold line |
| Frequency Distribution | Alerts plotted by frequency with UAS band shading |
| Latency Histogram | Processing latency with 2000ms budget line |
| Severity Donut | HIGH / MEDIUM / LOW alert breakdown |
| Alert Table | 40 most recent alerts with colour-coded badges |
| Evidence Snapshot | Full metadata of latest UAS-LIKE detection |
| Signature Library | All stored RF fingerprints with occurrence counts |
| Pipeline Performance | Throughput, latency stats, budget compliance |

---

## Supported Input Formats

| Format | Description |
|---|---|
| `.npz` | NumPy spectrogram bundle (recommended for testing) |
| `.sigmf-meta` | SigMF standard RF dataset |
| `.csv` | Comma-separated spectrogram matrix |
| `.png / .jpg` | Grayscale image treated as spectrogram |
| Directory | Folder of any of the above formats |

---

## Detection Pipeline
Input File
↓
Parser (SigMF / Spectrogram)
↓
Feature Extractor (28 features)
↓
Ensemble Classifier
├── Random Forest      (45%)
├── Gradient Boosting  (35%)
└── Rule-based Oracle  (20%)
↓
Alert Engine (≤ 2000ms latency budget)
↓
Signature Library Update
↓
Dashboard (auto-refresh every 5s)

---

## ML Model Details

| Component | Specification |
|---|---|
| Random Forest | 200 trees, max depth 12, balanced class weights |
| Gradient Boosting | 150 trees, max depth 5 |
| Calibration | Platt scaling (3-fold CV) |
| Target Pd | ≥ 0.90 |
| Target FAR | ≤ 0.03 |
| Target F1 | ≥ 0.85 |

---

## Requirements
Python 3.8+
numpy
pandas
scikit-learn
scipy
matplotlib
dash
plotly

Install all with:
```bash
pip install -r requirements.txt
```

---

