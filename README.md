# Traffic Demand Prediction using LightGBM & XGBoost
**Innovexa Catalyst - Machine Learning Project Task**

This repository contains a complete, production-ready machine learning system designed to predict traffic demand (volume) along the Interstate 94 corridor using historical traffic, environmental, and meteorological data.

---

## 1. Project Overview & Objective
The goal is to develop a high-accuracy machine learning regressor to forecast hourly traffic volume. Using the Metro Interstate Traffic Volume dataset, we engineered time-based, calendar, and weather impact features. By combining LightGBM and XGBoost in a weighted ensemble model, the system achieves an **R² score of 97.89%** and a **Mean Absolute Percentage Error (MAPE) of 0.54%** on a holdout test set. 

This model is deployed as an interactive Streamlit application containing predictive dashboards, 3D geospatial maps, travel time route planners, accident hazard risk ratings, and live sensor stream simulators.

---

---

## 2. Step-by-Step Execution Instructions

### Step 1: Clone and Set Up Environment
Ensure Python 3.8+ is installed. Navigate to the project directory and create a virtual environment, then install requirements.
```bash
# Navigate to project directory
cd d:/INNOVEXA/PROJECT-3

# Create a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Train Models & Generate Visuals
Execute the training pipeline to perform cleaning, feature engineering, Optuna hyperparameter optimization, model ensembling, evaluation, and visualization exports.
```bash
python train_models.py
```

### Step 3: Compile the PDF Case Study Report
Run the PDF generator script to compile the ReportLab document incorporating the statistical tables and visualization plots.
```bash
python generate_pdf_report.py
```

### Step 4: Run the Streamlit Dashboard App
Launch the interactive web application locally to interact with predictions, map visualizations, route recommendations, and live detector streams.
```bash
streamlit run app.py
```

---

## 3. Verification Table (Requirements vs. Implementation)

| PDF Requirement | Implemented File / Solution | Details / Achievement |
| :--- | :--- | :--- |
| **Exploratory Data Analysis** | `train_models.py`, `Traffic_Demand_Prediction.ipynb` | Missing values checked, distribution plots generated, correlation matrices computed. |
| **Outlier Detection & Cleaning** | `train_models.py` | Imputed 0.0 Kelvin values (median 282.46K); removed extreme rain outlier (9,831.3 mm). |
| **Feature Engineering** | `train_models.py` | Added `IsWeekend`, `IsPeakHour`, `IsRushHour`, `WeatherImpactScore`, and `traffic_density_score`. |
| **Model Comparison** | `train_models.py` | Compares Random Forest, Tuned XGBoost, Tuned LightGBM, and the Ensemble model. |
| **Optuna Tuning** | `train_models.py` | 10-trial Bayesian search on XGBoost and LightGBM using K-Fold CV. |
| **Ensemble Specification** | `train_models.py` | Blended Ensemble model: 55% LightGBM + 45% XGBoost. |
| **Target R² Evaluation** | `model_evaluation_metrics.csv` | Random Forest: **96.37%**  <br/>XGBoost: **97.87%**  <br/>LightGBM: **97.82%**  <br/>Ensemble: **97.89%** |
| **Streamlit Deployment** | `app.py` | Features: Location/Road dropdowns, Weather sliders, 24-hr Forecast, Peak Alerts, Route Advising. |
| **Jupyter Notebook** | `Traffic_Demand_Prediction.ipynb` | Full runnable analysis notebook containing the complete workflow. |
| **Professional PDF Report** | `Traffic_Demand_Prediction_Report.pdf` | A publication-grade ReportLab PDF study with embedded plots and metrics. |
| **Accident Risk Rating** | `app.py` | Computes hazard probability from weather severity and predicted traffic. |
| **Route Recommendations** | `app.py` | Compares A, B, and C routes, advising bypass options based on traffic volume. |
| **3D Geospatial Map** | `app.py` | Renders a Pydeck map along the Minneapolis I-94 corridor with live traffic densities. |
| **Presentation & Video Assets** | `PPT_Presentation_Outline.md`, `Demo_Video_Script.md` | Provides ready-to-use PPT slides structure and demo recording narration script. |

---

## 4. Submission Checklist

- [x] **Source Code**: Fully functional `train_models.py`, `generate_pdf_report.py`, and `app.py`.
- [x] **Runnable Notebook**: `Traffic_Demand_Prediction.ipynb` with all outputs saved.
- [x] **Trained Model Files**: Saved `.pkl` files for scaler, LightGBM, XGBoost, and the Ensemble model.
- [x] **Visualizations**: Six exported charts (`.png`) documenting target skew, hourly trends, correlations, importances, and fits.
- [x] **Evaluation Summary Table**: Saved `model_evaluation_metrics.csv` with holdout test metrics.
- [x] **Case Study Report**: Compiled, styled `Traffic_Demand_Prediction_Report.pdf`.
- [x] **Metadata Documentation**: Metadata details and preprocessing logs in `Dataset_Documentation.md`.
- [x] **Presentation Outline**: Outline for slides in `PPT_Presentation_Outline.md`.
- [x] **Recording Script**: Narration script for recording in `Demo_Video_Script.md`.
- [x] **Dependencies list**: Complete `requirements.txt`.
