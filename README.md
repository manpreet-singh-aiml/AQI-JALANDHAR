# FedLSTM-AQI: Federated Deep Learning for Air Quality Index Prediction

This repository contains the complete implementation of 
**FedLSTM-AQI**, a federated deep learning framework for 
privacy-preserving Air Quality Index (AQI) prediction, 
as described in the paper:

> **FedLSTM-AQI: A Federated Deep Learning Framework for 
> Air Quality Index Prediction**  
> Jaspal Kaur Saini, Manpreet Singh, Divya Bansal  
> *Soft Computing, Springer (Under Review)*

---

## Project Structure
AQI-JALANDHAR/
├── preprocess.py          # Timestamp correction + missing value interpolation
├── compute_aqi.py         # CPCB-standard AQI sub-index computation
├── lstm_training.py       # LSTM model — 5-seed training + sensor validation
├── bilstm_training.py     # BiLSTM + Attention — 5-seed training + sensor validation
├── federated_approach.py  # Federated Learning (FedAvg + Paillier HE)
└── requirements.txt

---

## Features

- Complete end-to-end AQI forecasting pipeline
- Chronological train-test split with leakage-free Min-Max scaling
- CPCB-based AQI sub-index computation (6 pollutants)
- LSTM and BiLSTM + Attention architectures with:
  - Layer Normalization
  - Residual connections
  - Dropout
  - Adam optimizer with gradient clipping
- Statistical reliability: 5 independent runs (seeds: 42, 123, 256, 789, 1024)
- Results reported as mean ± std with paired t-test significance
- Federated Learning with 3 genuinely heterogeneous clients:
  - Client 1: CPCB monitoring station (2017–2023, ~51,864 records)
  - Client 2: Airveda outdoor IoT sensor (Jun–Sep, all 6 features)
  - Client 3: Airveda indoor IoT sensor (Jun–Sep, PM2.5 + PM10 only)
- Structural zero-padding for feature-incomplete indoor client
- Sample-proportional FedAvg aggregation
- Paillier homomorphic encryption on output Dense layer
- Evaluation metrics: RMSE, MAE, RMSLE, MAPE, R²
- Both normalized and denormalized (AQI scale) reporting

---

## Installation

```bash
git clone https://github.com/SINGH-MANPREET-1708/AQI-JALANDHAR.git
cd AQI-JALANDHAR
pip install -r requirements.txt
```

---

## Dataset

The CPCB training dataset is **not included** in this 
repository due to size constraints. Download it from:

- [CPCB AQI Repository](https://airquality.cpcb.gov.in/ccr/)
- [Kaggle Mirror](https://www.kaggle.com/datasets/abhisheksjha/time-series-air-quality-data-of-india-2010-2023)

Filter for **Jalandhar, Punjab** after download.

Sensor datasets (Airveda outdoor + indoor) were collected 
at NIT Jalandhar campus and are available from the 
corresponding author upon reasonable request.

---

## Usage

### Step 1 — Preprocess raw CPCB data
```bash
python preprocess.py
```
Input: `jld_aqi.csv`  
Output: `jld_aqi_filled.csv`

### Step 2 — Compute AQI
```bash
python compute_aqi.py
```
Input: `jld_aqi_filled.csv`  
Output: `jld_aqi_with_aqi.csv`

### Step 3 — Train LSTM (centralized baseline)
```bash
python lstm_training.py
```
Trains over 5 seeds, evaluates on CPCB test set and 
Airveda sensor data.

### Step 4 — Train BiLSTM + Attention (centralized baseline)
```bash
python bilstm_training.py
```
Same evaluation protocol as LSTM.

### Step 5 — Federated Learning with Paillier HE
```bash
python federated_approach.py
```
Runs FedAvg over 5 rounds with 3 real heterogeneous 
clients. Paillier homomorphic encryption applied to 
output Dense layer. Evaluates both LSTM and BiLSTM 
global models.

---

## Requirements
numpy==1.23.5
pandas==1.5.3
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.2.2
tensorflow==2.12.0
phe==1.5.0

---

## Citation

If you use this code, please cite:
Saini, J.K., Singh, M., Bansal, D. (2025).
FedLSTM-AQI: A Federated Deep Learning Framework
for Air Quality Index Prediction.
Soft Computing, Springer. (Under Review)

---

## Contact

**Manpreet Singh**  
B.Tech CSE (AI & ML)  
DAV Institute of Engineering & Technology, Jalandhar  
mrsingh31524@gmail.com