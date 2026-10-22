# 🌊 Cloud-Based Flood Risk Prediction & Early Warning System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational, beginner-friendly Machine Learning and Cloud Web Application designed to predict regional flood inundation risk and provide early warning advisories. The application is built with **scikit-learn** and **Streamlit**, and is fully compatible with free **Streamlit Community Cloud** hosting.

> **⚠️ IMPORTANT EDUCATIONAL NOTICE & DISCLAIMER:**  
> This software is created **strictly for educational demonstration and research purposes**. It generates statistical probability estimates using historical meteorological records and does **NOT** constitute an official hydrological warning. Real-world flood emergencies depend on dynamic river channel discharge, upstream basin rainfall, storm surges, and drainage infrastructure. Always follow official life-safety warnings issued by national disaster authorities.

---

## 📌 Project Overview

Floods represent one of the most devastating natural hazards worldwide. This project demonstrates how modern cloud-native machine learning workflows can translate historical environmental measurements into actionable, three-tier risk advisories:

- **Low Risk (Normal Baseline):** Probability $< 25\%$; empirical historical flood incidence rate is $\approx 1.0\%$.
- **Medium Risk (Watch Advisory):** Probability $25\% - 60\%$; empirical historical flood incidence rate rises to $\approx 24.5\%$.
- **High Risk (Warning Level):** Probability $\ge 60\%$; empirical historical flood incidence rate surges to $\approx 89.2\%$.

The project incorporates end-to-end data validation, pipeline artifact serialization, interactive web dashboards, preset climate scenarios, batch CSV processing, and automated unit test suites.

---

## 🗂️ Project Directory Structure

```text
cloud_work/
├── .gitignore                      # Git exclusion rules for virtual environments and cache
├── README.md                       # Comprehensive setup, usage, and deployment documentation
├── requirements.txt                # Pinned dependencies for local execution and Streamlit Cloud
├── app.py                          # Streamlit web dashboard application (main cloud entry point)
│
├── data/
│   ├── README.md                   # Dataset provenance, schema definitions, and citation details
│   ├── download_data.py            # Reproducible download and integrity verification script
│   └── raw/
│       └── FloodPrediction.csv     # Historical meteorological dataset (20,544 observation records)
│
├── src/
│   ├── __init__.py                 # Python package initializer
│   ├── preprocess.py               # Schema definitions, physical input validation, and scikit-learn transformers
│   ├── train.py                    # Training script: fits Pipeline, evaluates test set, and exports artifacts
│   ├── evaluate.py                 # Classification metrics, confusion matrix, and calibration calculations
│   └── predict.py                  # Inference engine: handles single/batch inputs, risk tiers, and driver factors
│
├── models/
│   ├── flood_risk_pipeline.joblib  # Serialized scikit-learn pipeline (ColumnTransformer + RandomForest)
│   ├── evaluation_metrics.json     # Pre-computed test metrics (accuracy, precision, recall, F1, ROC-AUC)
│   └── feature_metadata.json       # Feature bounds, defaults, units, and risk tier configuration
│
└── tests/
    ├── __init__.py                 # Test package initializer
    └── test_model_pipeline.py      # Unit and integration test suite (validation, inference, boundaries)
```

### Purpose of Every File

| File | Purpose |
| :--- | :--- |
| [`app.py`](file:///Users/ravritroshan/Downloads/cloud_work/app.py) | Main interactive web application powered by Streamlit. Provides the dashboard with sliders, preset climate scenarios, batch CSV testing, evaluation visualizations, and safety guidance. |
| [`requirements.txt`](file:///Users/ravritroshan/Downloads/cloud_work/requirements.txt) | Minimal, pinned Python library requirements. Enables single-click deployment on free Streamlit Community Cloud. |
| [`data/download_data.py`](file:///Users/ravritroshan/Downloads/cloud_work/data/download_data.py) | Automates downloading the open historical meteorological dataset from public repositories with error handling and integrity checks. |
| [`data/README.md`](file:///Users/ravritroshan/Downloads/cloud_work/data/README.md) | Fully documents the historical dataset, IEEE academic paper citation, ground truth sources, and feature units. |
| [`src/preprocess.py`](file:///Users/ravritroshan/Downloads/cloud_work/src/preprocess.py) | Centralizes input boundaries, physical sanity checks (e.g. $T_{\min} \le T_{\max}$, non-negative precipitation), and scikit-learn `ColumnTransformer` pipelines. |
| [`src/train.py`](file:///Users/ravritroshan/Downloads/cloud_work/src/train.py) | Reproducible training pipeline using stratified 80/20 train/test split. Fits a balanced `RandomForestClassifier` and exports project artifacts. |
| [`src/evaluate.py`](file:///Users/ravritroshan/Downloads/cloud_work/src/evaluate.py) | Evaluates accuracy, precision, recall, F1, ROC-AUC, Brier score, confusion matrix, and empirical risk tier calibration. |
| [`src/predict.py`](file:///Users/ravritroshan/Downloads/cloud_work/src/predict.py) | Production inference module. Validates inputs, produces calibrated probabilities, assigns risk tiers, identifies top driving factors, and formats actionable advisories. |
| [`models/flood_risk_pipeline.joblib`](file:///Users/ravritroshan/Downloads/cloud_work/models/flood_risk_pipeline.joblib) | Serialized scikit-learn Pipeline artifact containing both the fitted feature scaler/imputer and the ensemble tree classifier. |
| [`models/evaluation_metrics.json`](file:///Users/ravritroshan/Downloads/cloud_work/models/evaluation_metrics.json) | Exported test performance metrics for instant rendering in the dashboard without re-evaluation latency. |
| [`models/feature_metadata.json`](file:///Users/ravritroshan/Downloads/cloud_work/models/feature_metadata.json) | Configuration JSON storing feature schemas, units, and risk tier threshold boundaries. |
| [`tests/test_model_pipeline.py`](file:///Users/ravritroshan/Downloads/cloud_work/tests/test_model_pipeline.py) | Unit test suite covering valid/invalid inputs, edge cases, risk tier boundaries, and batch processing. |

---

## 📊 Dataset Provenance & Academic Citation

This project does **not** invent training data or use unverified synthetic numbers. It trains on a genuine, peer-reviewed open meteorological dataset:

> **Academic Citation:**  
> Gauhar, N., Das, S., & Moury, K. S. (2021).  
> *"Prediction of Flood in Bangladesh using k-Nearest Neighbors Algorithm."*  
> **2021 2nd International Conference on Robotics, Electrical and Signal Processing Techniques (ICREST)**, pp. 357–361. IEEE.  
> Open Repository: [https://github.com/n-gauhar/Flood-prediction](https://github.com/n-gauhar/Flood-prediction)

### Key Dataset Statistics
- **Temporal Coverage:** 1949 – 2015 (over 65 years of continuous observations).
- **Spatial Coverage:** 35 regional meteorological stations.
- **Total Records:** 20,544 station-month records.
- **Ground Truth Inundation Records:** Verified against official annual disaster archives from the **Flood Forecasting and Warning Centre (FFWC)**, the **Bangladesh Water Development Board (BWDB)**, UN OCHA ReliefWeb, and the Dartmouth Flood Observatory.

### Supported Input Features & Physical Units

| Parameter | Unit | Allowable Range | Climatological Context |
| :--- | :--- | :--- | :--- |
| `Month` | 1 – 12 | 1 to 12 | Calendar month (monsoon peaks June–September) |
| `Rainfall` | mm | $0.0 - 2500.0$ | Total monthly precipitation recorded at monitoring station |
| `Relative_Humidity` | % | $10.0 - 100.0$ | Relative humidity percentage (values $>80\%$ indicate saturated air) |
| `Cloud_Coverage` | Oktas | $0.0 - 8.0$ | Cloud cover in eighths of sky (0 = clear, 8 = completely overcast) |
| `Bright_Sunshine` | Hours/day | $0.0 - 14.0$ | Daily average sunshine (decreases sharply during rainy spells) |
| `ALT` | Meters | $0.0 - 500.0$ | Elevation above mean sea level (lowlands face severe drainage delays) |
| `Wind_Speed` | m/s | $0.0 - 35.0$ | Average wind speed at station |
| `Max_Temp` | °C | $15.0 - 50.0$ | Maximum monthly temperature recorded |
| `Min_Temp` | °C | $0.0 - 35.0$ | Minimum monthly temperature recorded |

---

## 📈 Model Performance & Risk Tier Justification

The model pipeline is evaluated on a held-out test split of **4,109 samples** ($20\%$ stratified test split):

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Overall Accuracy** | **95.30%** | Reliable discrimination between flood and non-flood months |
| **Precision (Flood Class)** | **85.36%** | High certainty when warning of flood conditions |
| **Recall (Sensitivity)** | **92.49%** | Identifies $>92\%$ of historical inundation events |
| **F1-Score (Flood Class)** | **88.79%** | Harmonic balance between precision and sensitivity |
| **ROC-AUC Score** | **0.9910** | Exceptional ranking discrimination across decision thresholds |
| **Brier Score Loss** | **0.0318** | High probability calibration fidelity |

### Confusion Matrix (Test Split)
- **True Negatives ($TN$):** 3,152 (Correctly identified non-flood conditions)
- **False Positives ($FP$):** 131 (False alarms)
- **False Negatives ($FN$):** 62 (Missed flood events)
- **True Positives ($TP$):** 764 (Accurately warned inundation events)

### Empirical Risk Tier Justification

Rather than arbitrary percentage cutoffs, the early warning system's three tiers are **empirically calibrated** against actual historical flood outcomes in the test set:

| Risk Tier | Probability Range | Test Samples | Share of Test Set | Mean Model Prob | Empirical Flood Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🟢 **Low Risk (Normal)** | $0\% - 25\%$ | 3,077 | $74.9\%$ | $2.3\%$ | **$1.0\%$** |
| 🟡 **Medium Risk (Advisory)** | $25\% - 60\%$ | 192 | $4.7\%$ | $42.1\%$ | **$24.5\%$** |
| 🔴 **High Risk (Warning)** | $60\% - 100\%$ | 840 | $20.4\%$ | $92.1\%$ | **$89.2\%$** |

This confirms that when the model assigns **Low Risk**, the actual historical flood occurrence was only $1.0\%$. Conversely, when assigned **High Risk**, $89.2\%$ of records corresponded to verified historical flooding.

---

## 🛡️ Input Validation & Graceful Error Handling

The application protects against invalid or physically impossible inputs at multiple stages:
1. **Physical Consistency Checks:** Verifies that Minimum Temperature does not exceed Maximum Temperature ($T_{\min} \le T_{\max}$).
2. **Domain Boundary Enforcement:** Ensures rainfall is non-negative ($Rainfall \ge 0$), relative humidity is bounded $[10\%, 100\%]$, cloud coverage is bounded $[0, 8]$ oktas, and month is within $[1, 12]$.
3. **Graceful UI Feedback:** Invalid inputs trigger structured warning banners with specific explanatory messages rather than crashing the application or returning misleading predictions.

---

## 💻 Local Setup & Execution Guide

Follow these exact terminal commands to install and test the project locally.

### 1. Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12 installed on your system.
- `git` installed.

### 2. Clone Repository & Navigate to Directory
```bash
git clone <your-repository-url>
cd cloud_work
```

### 3. Create and Activate Virtual Environment
On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows (Command Prompt / PowerShell):
```cmd
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Download Dataset (Automated)
```bash
python data/download_data.py
```
*Note: This downloads `FloodPrediction.csv` into `data/raw/` and verifies row integrity.*

### 6. Train the Machine Learning Pipeline & Export Artifacts
```bash
python src/train.py
```
*Expected Output:*
- Trains `RandomForestClassifier` on 16,435 samples.
- Saves pipeline to `models/flood_risk_pipeline.joblib`.
- Saves evaluation report to `models/evaluation_metrics.json`.
- Prints test summary table (Accuracy $\approx 95.3\%$, ROC-AUC $\approx 0.991$).

### 7. Run Unit and Integration Test Suite
```bash
python -m unittest tests/test_model_pipeline.py
```
*All 9 unit tests should pass with `OK` in under a second.*

### 8. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## ☁️ Free Streamlit Community Cloud Deployment Guide

This project is configured specifically for 100% free hosting on **Streamlit Community Cloud** without requiring credentials, API keys, or specialized cloud infrastructure:

1. **Push Code to GitHub:**
   Ensure your local project is pushed to a public or private GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Cloud-Based Flood Risk Prediction System"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
   *(Note: The pre-trained `models/flood_risk_pipeline.joblib` artifact is only ~6.8 MB, comfortably under GitHub's 100 MB file limit).*

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
   - Click **"New app"**.
   - Select your repository, branch (`main`), and set the **Main file path** to `app.py`.
   - Click **"Deploy!"**.

3. **Automatic Cloud Build:**
   Streamlit Community Cloud automatically reads `requirements.txt`, provisions the Python runtime, and starts the dashboard in under 2 minutes.

---

## 🧑‍🏫 Educational Guidance & Official Resources

Floods are among the most dynamic natural hazards on earth. While statistical machine learning models provide powerful situational estimates based on precipitation and humidity patterns, they **cannot account for upstream dam operations, dike breaches, or tidal surges**.

For official, operational flood warnings, always consult:
- **Bangladesh Water Development Board (BWDB FFWC):** [http://www.ffwc.gov.bd](http://www.ffwc.gov.bd)
- **India Meteorological Department (IMD Flood Meteorological Office):** [https://mausam.imd.gov.bd](https://mausam.imd.gov.bd)
- **US National Oceanic and Atmospheric Administration (NOAA NWS):** [https://water.weather.gov](https://water.weather.gov)
- **UK Environment Agency Flood Warning Service:** [https://check-for-flooding.service.gov.uk](https://check-for-flooding.service.gov.uk)
- **Global Flood Awareness System (GloFAS - Copernicus):** [https://www.globalfloods.eu](https://www.globalfloods.eu)

---

## 📄 License
This educational project is released under the **MIT License**.
