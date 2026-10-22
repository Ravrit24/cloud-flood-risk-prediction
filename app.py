"""
Cloud-Based Flood Risk Prediction & Early Warning System.

A beginner-friendly Streamlit web application powered by scikit-learn.
Compatible with free Streamlit Community Cloud deployment.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocess import (
    FEATURE_COLUMNS,
    FEATURE_SPECS,
    validate_input_data
)
from src.predict import (
    load_trained_pipeline,
    load_evaluation_metrics,
    predict_flood_risk,
    predict_batch
)
from src.evaluate import RISK_TIERS

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Flood Risk Early Warning System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    /* Disclaimer Alert Box */
    .disclaimer-box {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        color: #991b1b;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    
    /* Result Cards */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin: 0.4rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
    }
    
    /* Risk Badge */
    .badge-pill {
        display: inline-block;
        padding: 0.45rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.03em;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .factor-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Load and cache the trained scikit-learn pipeline."""
    try:
        return load_trained_pipeline()
    except Exception as e:
        st.error(f"Error loading trained model pipeline: {e}")
        st.info("Tip: If running for the first time, run 'python src/train.py' in your terminal.")
        return None


@st.cache_data(show_spinner=False)
def get_metrics():
    """Load and cache pre-computed evaluation metrics."""
    return load_evaluation_metrics()


# Preset Demonstration Scenarios
PRESET_SCENARIOS = {
    "Select a Preset Scenario...": None,
    "1. Dry Winter Season (Jan - Low Risk)": {
        "Month": 1,
        "Rainfall": 12.0,
        "Relative_Humidity": 64.0,
        "Cloud_Coverage": 1.2,
        "Bright_Sunshine": 8.5,
        "ALT": 12.0,
        "Wind_Speed": 1.5,
        "Max_Temp": 29.5,
        "Min_Temp": 14.0
    },
    "2. Moderate Pre-Monsoon Showers (May - Medium Risk)": {
        "Month": 5,
        "Rainfall": 230.0,
        "Relative_Humidity": 78.0,
        "Cloud_Coverage": 4.8,
        "Bright_Sunshine": 5.8,
        "ALT": 8.0,
        "Wind_Speed": 2.8,
        "Max_Temp": 34.0,
        "Min_Temp": 24.5
    },
    "3. Severe Monsoon Surge (July - High Risk)": {
        "Month": 7,
        "Rainfall": 620.0,
        "Relative_Humidity": 89.0,
        "Cloud_Coverage": 6.8,
        "Bright_Sunshine": 2.4,
        "ALT": 5.0,
        "Wind_Speed": 3.4,
        "Max_Temp": 32.5,
        "Min_Temp": 26.0
    },
    "4. Low-Lying Coastal Storm (August - High Risk)": {
        "Month": 8,
        "Rainfall": 510.0,
        "Relative_Humidity": 91.0,
        "Cloud_Coverage": 6.5,
        "Bright_Sunshine": 3.0,
        "ALT": 3.0,
        "Wind_Speed": 4.2,
        "Max_Temp": 31.8,
        "Min_Temp": 25.5
    }
}


# ---------------------------------------------------------
# Sidebar Controls & Scenario Selection
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/water.png", width=64)
st.sidebar.title("System Controls")

st.sidebar.markdown("### Quick Preset Scenarios")
selected_preset_name = st.sidebar.selectbox(
    "Load typical climatological scenarios:",
    list(PRESET_SCENARIOS.keys()),
    index=0,
    help="Select a benchmark scenario to automatically populate all environmental sliders."
)

preset_values = PRESET_SCENARIOS.get(selected_preset_name)

st.sidebar.markdown("---")
st.sidebar.markdown("### About This System")
st.sidebar.info(
    "**Project:** Cloud-Based Flood Risk Prediction & Early Warning System\n\n"
    "**Model:** Random Forest Classifier with scikit-learn Pipeline\n\n"
    "**Dataset:** 65+ Years Historical Weather & Flood Archive (Gauhar et al., IEEE 2021)\n\n"
    "**Deployment:** Free Streamlit Community Cloud"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "⚠️ **Educational Notice:** Predictions are statistical estimates and not official emergency warnings."
)


# ---------------------------------------------------------
# Main Page Header & Disclaimer Banner
# ---------------------------------------------------------
st.markdown('<h1 class="main-header">🌊 Cloud-Based Flood Risk Prediction & Early Warning System</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Interactive Machine Learning Early Warning Assessment '
    'Trained on Historical Meteorological & Inundation Records</p>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ IMPORTANT EDUCATIONAL NOTICE & DISCLAIMER:</strong><br>
    This application is an educational and technical demonstration tool utilizing a machine learning classification pipeline.
    It produces <em>statistical estimates</em> based on historical meteorological correlations. <strong>It is NOT an official hydrological warning.</strong>
    Real-world flooding is influenced by complex dynamic factors such as river basin discharge, tidal surges, dam spillway operations, and local urban drainage infrastructure.
    For life-safety decisions, always refer to your local or national meteorological and disaster management authorities.
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_predict, tab_batch, tab_metrics, tab_about = st.tabs([
    "🎯 Early Warning Assessment",
    "📁 Batch Evaluation",
    "📊 Model Performance & Justification",
    "📖 About Dataset & Science"
])

pipeline = get_pipeline()
metrics_data = get_metrics()


# =========================================================
# TAB 1: Early Warning Assessment (Single Prediction)
# =========================================================
with tab_predict:
    st.markdown("### 1. Enter Environmental & Meteorological Parameters")
    st.write(
        "Adjust the sliders or input boxes below to evaluate flood risk for specific meteorological conditions. "
        "Values can be manually tweaked or auto-populated using the sidebar presets."
    )

    # Form with columns
    with st.form("risk_assessment_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 🌧️ Precipitation & Moisture")
            def_month = preset_values["Month"] if preset_values else 7
            month_input = st.selectbox(
                "Calendar Month",
                options=list(range(1, 13)),
                index=def_month - 1,
                format_func=lambda m: [
                    "1 - January (Winter Dry)",
                    "2 - February (Winter Dry)",
                    "3 - March (Pre-Monsoon)",
                    "4 - April (Pre-Monsoon)",
                    "5 - May (Pre-Monsoon Showers)",
                    "6 - June (Monsoon Onset)",
                    "7 - July (Monsoon Peak)",
                    "8 - August (Monsoon Peak)",
                    "9 - September (Monsoon Withdrawal)",
                    "10 - October (Post-Monsoon)",
                    "11 - November (Dry Season)",
                    "12 - December (Winter Dry)"
                ][m - 1],
                help="Historical flood events strongly cluster during the South Asian summer monsoon (June-September)."
            )

            def_rain = preset_values["Rainfall"] if preset_values else 350.0
            rainfall_input = st.number_input(
                "Monthly Precipitation (Rainfall in mm)",
                min_value=0.0,
                max_value=2500.0,
                value=float(def_rain),
                step=10.0,
                help="Total monthly rainfall in millimeters recorded by the station."
            )

            def_hum = preset_values["Relative_Humidity"] if preset_values else 82.0
            humidity_input = st.slider(
                "Relative Humidity (%)",
                min_value=10.0,
                max_value=100.0,
                value=float(def_hum),
                step=1.0,
                help="Average relative atmospheric humidity. Sustained values above 80% indicate saturated air."
            )

        with col2:
            st.markdown("##### ☁️ Atmospheric & Terrain")
            def_cloud = preset_values["Cloud_Coverage"] if preset_values else 5.5
            cloud_input = st.slider(
                "Cloud Coverage (Oktas, 0 - 8)",
                min_value=0.0,
                max_value=8.0,
                value=float(def_cloud),
                step=0.1,
                help="Fraction of celestial dome covered by clouds in eighths (0 = clear sky, 8 = overcast)."
            )

            def_sun = preset_values["Bright_Sunshine"] if preset_values else 4.2
            sun_input = st.slider(
                "Bright Sunshine Duration (Hours / Day)",
                min_value=0.0,
                max_value=14.0,
                value=float(def_sun),
                step=0.1,
                help="Average daily sunshine hours. Low sunshine correlates strongly with prolonged storms."
            )

            def_alt = preset_values["ALT"] if preset_values else 10.0
            alt_input = st.number_input(
                "Station Altitude / Elevation (Meters)",
                min_value=0.0,
                max_value=500.0,
                value=float(def_alt),
                step=1.0,
                help="Topographical elevation above sea level. Low elevations are vulnerable to inundation."
            )

        with col3:
            st.markdown("##### 🌡️ Temperature & Wind")
            def_wind = preset_values["Wind_Speed"] if preset_values else 2.5
            wind_input = st.slider(
                "Average Wind Speed (m/s)",
                min_value=0.0,
                max_value=35.0,
                value=float(def_wind),
                step=0.1,
                help="Mean wind speed recorded at the monitoring station."
            )

            def_max_temp = preset_values["Max_Temp"] if preset_values else 33.5
            max_temp_input = st.number_input(
                "Maximum Temperature (°C)",
                min_value=15.0,
                max_value=50.0,
                value=float(def_max_temp),
                step=0.5,
                help="Peak monthly temperature."
            )

            def_min_temp = preset_values["Min_Temp"] if preset_values else 24.5
            min_temp_input = st.number_input(
                "Minimum Temperature (°C)",
                min_value=0.0,
                max_value=35.0,
                value=float(def_min_temp),
                step=0.5,
                help="Lowest monthly temperature."
            )

        submit_btn = st.form_submit_button("⚡ Run Early Warning Assessment", use_container_width=True)

    # Prediction Execution
    input_payload = {
        "Month": month_input,
        "Rainfall": rainfall_input,
        "Relative_Humidity": humidity_input,
        "Cloud_Coverage": cloud_input,
        "Bright_Sunshine": sun_input,
        "ALT": alt_input,
        "Wind_Speed": wind_input,
        "Max_Temp": max_temp_input,
        "Min_Temp": min_temp_input
    }

    if submit_btn or preset_values is not None:
        if pipeline is None:
            st.error("Model pipeline is not loaded. Please train the model first by running `python src/train.py`.")
        else:
            # Validate and predict
            result = predict_flood_risk(input_payload, pipeline=pipeline)

            if not result["is_valid"]:
                st.error("⚠️ **Invalid Input Detected:**")
                for err in result["errors"]:
                    st.write(f"- {err}")
            else:
                st.markdown("---")
                st.markdown("### 2. Early Warning Prediction Results")

                tier = result["risk_tier"]
                prob_pct = result["flood_probability_pct"]
                color = result["risk_color"]
                badge_text = result["risk_badge"]

                # Metric Cards
                m_col1, m_col2, m_col3 = st.columns(3)

                with m_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Predicted Risk Tier</div>
                        <div class="badge-pill" style="background-color: {color};">{badge_text}</div>
                        <div style="font-size: 0.9rem; color: #475569;">Empirical Classification</div>
                    </div>
                    """, unsafe_allow_html=True)

                with m_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Inundation Probability</div>
                        <div class="metric-value" style="color: {color};">{prob_pct:.1f}%</div>
                        <div style="font-size: 0.9rem; color: #475569;">Continuous Model Estimate</div>
                    </div>
                    """, unsafe_allow_html=True)

                with m_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Binary Threshold Verdict</div>
                        <div class="metric-value" style="font-size: 1.6rem; margin-top: 0.8rem;">
                            {result['binary_label']}
                        </div>
                        <div style="font-size: 0.9rem; color: #475569;">Decision Boundary: P ≥ 0.50</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Visual Progress Gauge Bar
                st.markdown("##### 📈 Risk Probability Gauge")
                st.progress(result["flood_probability"])
                gauge_cols = st.columns(3)
                gauge_cols[0].caption("🟢 **Low Risk Tier:** 0% – 25%")
                gauge_cols[1].caption("🟡 **Medium Risk Tier:** 25% – 60%")
                gauge_cols[2].caption("🔴 **High Risk Tier:** 60% – 100%")

                # Guidance & Actionable Next Steps
                st.markdown("##### 💡 Advisory & Preparedness Recommendations")
                if tier == "Low":
                    st.success(f"**Baseline Normal Conditions:** {result['guidance']}")
                elif tier == "Medium":
                    st.warning(f"**Elevated Risk Advisory:** {result['guidance']}")
                else:
                    st.error(f"**Severe Flood Threat Warning:** {result['guidance']}")

                st.info(f"**Action Recommendation:** {result['recommendation']}")

                # Contributing Factors Analysis
                st.markdown("##### 🔍 Key Climatological Driving Factors")
                factors = result["contributing_factors"]
                if factors:
                    for f in factors:
                        st.markdown(f"""
                        <div class="factor-card">
                            <strong>{f['feature']}</strong> — <em>{f['impact']}</em><br>
                            {f['message']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("No extreme environmental anomalies were triggered. All variables reside within normal ranges.")


# =========================================================
# TAB 2: Batch Evaluation
# =========================================================
with tab_batch:
    st.markdown("### 📁 Batch Flood Risk Assessment")
    st.write(
        "Evaluate multiple observation points or meteorological stations simultaneously. "
        "Upload a CSV containing the required environmental parameters, or test with historical samples."
    )

    # Sample batch generator
    if st.button("🧪 Load Sample Historical Records for Batch Test"):
        sample_batch = pd.DataFrame([
            {"Month": 1, "Rainfall": 15.0, "Relative_Humidity": 65.0, "Cloud_Coverage": 1.5, "Bright_Sunshine": 8.0, "ALT": 12.0, "Wind_Speed": 1.2, "Max_Temp": 28.0, "Min_Temp": 13.0, "Station": "Station_Winter_Baseline"},
            {"Month": 5, "Rainfall": 280.0, "Relative_Humidity": 82.0, "Cloud_Coverage": 5.2, "Bright_Sunshine": 4.8, "ALT": 7.0, "Wind_Speed": 2.9, "Max_Temp": 34.5, "Min_Temp": 25.0, "Station": "Station_PreMonsoon_Delta"},
            {"Month": 7, "Rainfall": 680.0, "Relative_Humidity": 92.0, "Cloud_Coverage": 7.1, "Bright_Sunshine": 1.9, "ALT": 4.0, "Wind_Speed": 3.8, "Max_Temp": 31.5, "Min_Temp": 26.2, "Station": "Station_Monsoon_Lowland"},
            {"Month": 8, "Rainfall": 490.0, "Relative_Humidity": 87.0, "Cloud_Coverage": 6.0, "Bright_Sunshine": 3.5, "ALT": 35.0, "Wind_Speed": 2.1, "Max_Temp": 32.0, "Min_Temp": 24.8, "Station": "Station_Highland_Upland"},
            {"Month": 11, "Rainfall": 35.0, "Relative_Humidity": 70.0, "Cloud_Coverage": 2.0, "Bright_Sunshine": 7.5, "ALT": 15.0, "Wind_Speed": 1.4, "Max_Temp": 30.0, "Min_Temp": 18.0, "Station": "Station_Autumn_Dry"}
        ])
        st.session_state["batch_data"] = sample_batch

    uploaded_file = st.file_uploader(
        "Upload Custom Environmental CSV",
        type=["csv"],
        help=f"Must contain columns: {', '.join(FEATURE_COLUMNS)}"
    )

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            st.session_state["batch_data"] = df_uploaded
            st.success(f"Successfully loaded CSV with {len(df_uploaded)} records.")
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")

    if "batch_data" in st.session_state and pipeline is not None:
        df_batch = st.session_state["batch_data"]
        st.markdown("#### Input Records Preview:")
        st.dataframe(df_batch, use_container_width=True)

        if st.button("🚀 Process Batch Predictions", use_container_width=True):
            try:
                df_results = predict_batch(df_batch, pipeline=pipeline)
                st.markdown("#### Prediction Output:")
                st.dataframe(
                    df_results.style.map(
                        lambda val: "background-color: #d1fae5" if val == "Low" else (
                            "background-color: #fef3c7" if val == "Medium" else (
                                "background-color: #fee2e2" if val == "High" else ""
                            )
                        ),
                        subset=["Risk_Tier"]
                    ),
                    use_container_width=True
                )

                # Download processed CSV
                csv_bytes = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results as CSV",
                    data=csv_bytes,
                    file_name="flood_risk_predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error processing batch: {e}")


# =========================================================
# TAB 3: Model Performance & Justification
# =========================================================
with tab_metrics:
    st.markdown("### 📊 Model Performance on Held-Out Test Set")
    st.write(
        "The model was evaluated using a stratified 80/20 train/test split on 20,544 historical observation records. "
        "All metrics below reflect performance strictly on the unseen test set (4,109 samples)."
    )

    if metrics_data and "summary" in metrics_data:
        summary = metrics_data["summary"]
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("Overall Accuracy", f"{summary['accuracy']*100:.1f}%")
        s_col2.metric("Flood Precision", f"{summary['precision_flood']*100:.1f}%")
        s_col3.metric("Flood Recall (Sensitivity)", f"{summary['recall_flood']*100:.1f}%")
        s_col4.metric("ROC-AUC Score", f"{summary['roc_auc']:.4f}")

        st.markdown("---")
        st.markdown("### ⚖️ Empirical Justification of Low, Medium, and High Risk Tiers")
        st.write(
            "Rather than arbitrary thresholds, the three warning tiers are calibrated against "
            "actual historical flood incidence rates observed in the test partition:"
        )

        tier_calib = metrics_data.get("risk_tier_calibration", {})
        calib_rows = []
        for tier_name, t_info in tier_calib.items():
            calib_rows.append({
                "Risk Tier": tier_name,
                "Probability Range": f"{t_info['threshold_range'][0]*100:.0f}% – {t_info['threshold_range'][1]*100:.0f}%",
                "Test Set Samples": t_info["count"],
                "Test Share (%)": f"{t_info['percentage']:.1f}%",
                "Mean Predicted Prob (%)": f"{t_info['mean_predicted_prob']*100:.1f}%",
                "Actual Historical Flood Rate (%)": f"{t_info['empirical_flood_rate']*100:.1f}%"
            })
        st.table(pd.DataFrame(calib_rows))

        st.markdown("""
        **Key Insights:**
        - **Low Risk ($P < 0.25$):** Historical flood rate is only **~1.0%**. Almost all samples in this band experienced no inundation.
        - **Medium Risk ($0.25 \\le P < 0.60$):** Historical flood rate rises to **~24.5%** (elevated well above the 20% regional baseline).
        - **High Risk ($P \\ge 0.60$):** Historical flood rate surges to **~89.2%**, capturing nearly all extreme flood episodes.
        """)

        st.markdown("---")
        c_col1, c_col2 = st.columns(2)

        with c_col1:
            st.markdown("#### 🗂️ Confusion Matrix (Test Set)")
            cm_dict = metrics_data.get("confusion_matrix", {})
            cm_matrix = cm_dict.get("matrix", [[0, 0], [0, 0]])
            cm_df = pd.DataFrame(
                cm_matrix,
                index=["Actual: No Flood (0)", "Actual: Flood (1)"],
                columns=["Predicted: No Flood (0)", "Predicted: Flood (1)"]
            )
            st.dataframe(cm_df, use_container_width=True)
            st.caption(
                f"True Negatives: {cm_dict.get('true_negative')}, False Positives: {cm_dict.get('false_positive')}, "
                f"False Negatives: {cm_dict.get('false_negative')}, True Positives: {cm_dict.get('true_positive')}"
            )

        with c_col2:
            st.markdown("#### 🌟 Feature Importances")
            feat_imps = metrics_data.get("feature_importances", [])
            if feat_imps:
                df_imps = pd.DataFrame(feat_imps).set_index("feature")
                st.bar_chart(df_imps["importance_pct"])
                st.caption("Percentage contribution of each environmental feature within the ensemble trees.")
    else:
        st.info("No saved evaluation metrics found. Please execute `python src/train.py` to generate metrics.")


# =========================================================
# TAB 4: About Dataset & Educational Purpose
# =========================================================
with tab_about:
    st.markdown("### 📖 About This Educational Project")
    st.markdown("""
    #### 1. Academic & Dataset Provenance
    This system is built using an authentic, openly available historical meteorological and flood dataset published in IEEE:
    
    > **Gauhar, N., Das, S., & Moury, K. S. (2021).**  
    > *"Prediction of Flood in Bangladesh using k-Nearest Neighbors Algorithm."*  
    > 2021 2nd International Conference on Robotics, Electrical and Signal Processing Techniques (ICREST), pp. 357–361. IEEE.  
    > Open Repository: [https://github.com/n-gauhar/Flood-prediction](https://github.com/n-gauhar/Flood-prediction)
    
    - **Time Period:** 1949 – 2015 (over 65 years of observations).
    - **Total Records:** 20,544 station-month records across 35 monitoring stations.
    - **Ground Truth Flood Verification:** Annual flood records cross-referenced with the **Flood Forecasting and Warning Centre (FFWC)**, the **Bangladesh Water Development Board (BWDB)**, UN OCHA ReliefWeb, and the **Dartmouth Flood Observatory**.
    
    #### 2. Why Machine Learning Models Are Educational Estimates
    In operational disaster management, flood forecasting requires:
    1. **Dynamic Hydraulic Flow Routing:** Tracking water velocity and cross-sectional channel flow through river networks.
    2. **Basin-Wide Upstream Precipitation:** River discharge often depends on rainfall hundreds of kilometers upstream in neighboring regions.
    3. **Tidal & Surge Dynamics:** Coastal high tides can impede river discharge (backwater effect), causing severe inundation even under moderate local rainfall.
    4. **Structural Failures:** Breached embankments, dam emergency releases, and blocked municipal storm drains cannot be anticipated by point-in-time meteorological observations alone.
    
    For these reasons, point-in-time machine learning predictions must **always be interpreted as educational estimates**, not definitive forecasts.
    
    #### 3. Official Flood Warning & Emergency Resources
    - **Bangladesh Water Development Board (FFWC):** [ffwc.gov.bd](http://www.ffwc.gov.bd)
    - **India Meteorological Department (IMD):** [mausam.imd.gov.bd](https://mausam.imd.gov.bd)
    - **US National Weather Service / NOAA:** [water.weather.gov](https://water.weather.gov)
    - **UK Environment Agency Flood Warnings:** [check-for-flooding.service.gov.uk](https://check-for-flooding.service.gov.uk)
    - **Global Flood Awareness System (GloFAS):** [globalfloods.eu](https://www.globalfloods.eu)
    """)
