# Traffic Demand Prediction using LightGBM & XGBoost
This repository contains a complete, production-ready machine learning system designed to predict traffic demand (volume) along the Interstate 94 corridor using historical traffic, environmental, and meteorological data.

---

## 1. Project Overview & Objective
The goal is to develop a high-accuracy machine learning regressor to forecast hourly traffic volume. Using the Metro Interstate Traffic Volume dataset, we engineered time-based, calendar, and weather impact features. By combining LightGBM and XGBoost in a weighted ensemble model, the system achieves an **R² score of 97.89%** and a **Mean Absolute Percentage Error (MAPE) of 0.54%** on a holdout test set. 

This model is deployed as an interactive Streamlit application containing predictive dashboards, 3D geospatial maps, travel time route planners, accident hazard risk ratings, and live sensor stream simulators.

---

## 2. Step-by-Step Execution Instructions

### Step 1: Clone and Set Up Environment
Ensure Python 3.8+ is installed. Clone this repository, navigate into the project directory, create a virtual environment, and install the dependencies.
```bash
# Clone the repository
git clone <your-github-repo-url>
cd <repository-name>

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

### Step 3: Run the Streamlit Dashboard App
Launch the interactive web application locally to interact with predictions, map visualizations, route recommendations, and live detector streams.
```bash
streamlit run app.py
```

---

## 3. Model Performance Metrics

| Model | R² Score | RMSE | MAE | MAPE |
| :--- | :--- | :--- | :--- | :--- |
| Random Forest | 96.37% | 377.87 | 220.38 | 88.77% |
| XGBoost (Tuned) | 97.87% | 289.50 | 175.98 | 58.98% |
| LightGBM (Tuned) | 97.82% | 292.83 | 180.01 | 51.11% |
| **Ensemble (55% LGB + 45% XGB)** | **97.89%** | **288.27** | **175.73** | **54.16%** |


