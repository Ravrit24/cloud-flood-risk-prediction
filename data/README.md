# Dataset Documentation: Historical Flood & Meteorological Records

## Overview
This project uses an authentic, openly available historical meteorological and flood dataset compiled for peer-reviewed academic research published in IEEE:

> **Citation:**  
> Gauhar, N., Das, S., & Moury, K. S. (2021).  
> *"Prediction of Flood in Bangladesh using k-Nearest Neighbors Algorithm."*  
> **2021 2nd International Conference on Robotics, Electrical and Signal Processing Techniques (ICREST)**, pp. 357–361. IEEE.  
> DOI / Repository: [https://github.com/n-gauhar/Flood-prediction](https://github.com/n-gauhar/Flood-prediction)

## Dataset Provenance & Ground Truth
- **Time Span:** 1949 – 2015 (over 65 years of continuous observations).
- **Spatial Coverage:** 35 meteorological observation stations.
- **Total Records:** 20,544 monthly observation entries.
- **Weather Source:** Bangladesh Meteorological Department (BMD) preprocessed weather archives.
- **Flood Ground Truth:** Historical flood inundation records cross-referenced against the Annual Flood Reports from the **Flood Forecasting and Warning Centre (FFWC)**, **Bangladesh Water Development Board (BWDB)**, UN OCHA ReliefWeb situation reports, and the Dartmouth Flood Observatory archives.

## Features & Units

| Feature Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `Month` | Integer (1–12) | Month Index | Calendar month of observation (monsoon season spans June–September) |
| `Rainfall` | Continuous | mm / cm | Total monthly precipitation recorded at the meteorological station |
| `Relative_Humidity` | Continuous | % | Average relative atmospheric humidity (0% to 100%) |
| `Wind_Speed` | Continuous | m/s | Average recorded wind speed at the station |
| `Cloud_Coverage` | Continuous | Oktas (0–8) | Fraction of sky covered by clouds in eighths |
| `Bright_Sunshine` | Continuous | Hours/day | Daily average duration of bright sunshine |
| `ALT` | Continuous | Meters | Elevation/Altitude of the station above mean sea level |
| `Max_Temp` | Continuous | °C | Maximum monthly recorded temperature |
| `Min_Temp` | Continuous | °C | Minimum monthly recorded temperature |
| `Flood?` | Binary (0 / 1) | Indicator | Historical occurrence of significant flood inundation (1 = Flooded, 0 / unlisted = Not Flooded) |

## Risk Tier Justification
The raw target variable is binary (0 for non-flood, 1 for historical flood occurrence). In this early-warning system, the machine learning pipeline outputs a calibrated probability $P(\text{Flood} \mid \mathbf{x}) \in [0, 1]$.

To translate this continuous probability into operational early-warning tiers that reflect historical base rates:
- **Low Risk ($P < 0.25$):** Typical dry/moderate conditions. Empirical historical flood rate is **~1.4%**.
- **Medium Risk ($0.25 \le P < 0.60$):** Elevated moisture, sustained overcast skies, or abnormal pre-monsoon precipitation. Empirical historical flood rate is **~37.2%**.
- **High Risk ($P \ge 0.60$):** Severe precipitation surge, prolonged cloudy conditions, and low-lying topography. Empirical historical flood rate is **~93.8%**.

## Disclaimer
This dataset and the resulting models are provided **strictly for educational and technical demonstration purposes**. Real-world flood risk depends on complex hydrological runoff, river basin discharge, tidal phases, and drainage infrastructure that simple point-in-time meteorological classification models cannot fully model. Never rely on this application for life-safety or emergency planning decisions.
