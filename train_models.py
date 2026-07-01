import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import optuna

# Set style for plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

class TrafficEnsembleRegressor:
    """
    Custom Ensemble Regressor combining LightGBM and XGBoost
    using specified weights: 55% LightGBM + 45% XGBoost.
    """
    def __init__(self, lgb_model, xgb_model, lgb_weight=0.55, xgb_weight=0.45):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.lgb_weight = lgb_weight
        self.xgb_weight = xgb_weight
        
    def predict(self, X):
        lgb_preds = self.lgb_model.predict(X)
        xgb_preds = self.xgb_model.predict(X)
        return (self.lgb_weight * lgb_preds) + (self.xgb_weight * xgb_preds)

def main():
    print("--- Phase 1: Data Loading & Preprocessing ---")
    
    # 1. Load data
    csv_path = "Dataset/Metro_Interstate_Traffic_Volume.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Please ensure the dataset exists.")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # 2. Outlier Cleansing
    initial_len = len(df)
    
    # Impute temp == 0.0 with median of valid temps
    valid_temp_median = df.loc[df["temp"] > 0, "temp"].median()
    zero_temp_count = (df["temp"] <= 0).sum()
    df.loc[df["temp"] <= 0, "temp"] = valid_temp_median
    print(f"Imputed {zero_temp_count} temperature records containing 0.0 Kelvin with median ({valid_temp_median:.2f} K).")
    
    # Remove rain_1h extreme outliers (> 100mm)
    extreme_rain_count = (df["rain_1h"] > 100).sum()
    df = df[df["rain_1h"] <= 100].copy()
    print(f"Removed {extreme_rain_count} extreme rain outliers (> 100mm). New shape: {df.shape}")
    
    # 3. Missing Value Handling
    # holiday column has many NaNs, representing non-holidays. Impute as 'None'
    df["holiday"] = df["holiday"].fillna("None")
    print("Handled missing values in 'holiday' by filling with 'None'.")
    
    # Parse date_time
    df["date_time"] = pd.to_datetime(df["date_time"])
    
    # Extract baseline date-time components
    df["Year"] = df["date_time"].dt.year
    df["Month"] = df["date_time"].dt.month
    df["Day"] = df["date_time"].dt.day
    df["Hour"] = df["date_time"].dt.hour
    df["DayOfWeek"] = df["date_time"].dt.dayofweek
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    
    # 4. Train-Test Split (80/20 split)
    # Splitting prior to feature scaling and target encoding to prevent data leakage
    X = df.drop(columns=["traffic_volume"])
    y = df["traffic_volume"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")
    
    # Save the original test labels and data for final evaluation
    test_eval_df = X_test.copy()
    test_eval_df["traffic_volume"] = y_test
    
    # 5. Feature Engineering
    print("Engineering features...")
    
    # Calculate Traffic Density Score: historical average traffic volume per Hour and DayOfWeek
    # Calculated ONLY on the training set to prevent data leakage, then mapped to test set.
    group_cols = ["Hour", "DayOfWeek"]
    train_density = pd.concat([X_train, y_train], axis=1).groupby(group_cols)["traffic_volume"].mean().reset_index()
    train_density.rename(columns={"traffic_volume": "traffic_density_score"}, inplace=True)
    
    # Convert density mapping to a dictionary for deployment
    density_mapping = train_density.set_index(["Hour", "DayOfWeek"])["traffic_density_score"].to_dict()
    train_mean_fallback = y_train.mean()
    
    def apply_density_score(data):
        res = data.copy()
        res = pd.merge(res, train_density, on=group_cols, how="left")
        res["traffic_density_score"] = res["traffic_density_score"].fillna(train_mean_fallback)
        return res
        
    X_train = apply_density_score(X_train)
    X_test = apply_density_score(X_test)
    
    # Define adding flags and scores
    def add_engineered_features(data):
        df_feat = data.copy()
        # Weekend Flag
        df_feat["IsWeekend"] = (df_feat["DayOfWeek"] >= 5).astype(int)
        # Peak Hour Flag (Weekday morning 7 AM, weekday evening 4 PM)
        df_feat["IsPeakHour"] = ((df_feat["IsWeekend"] == 0) & df_feat["Hour"].isin([7, 16])).astype(int)
        # Rush Hour Indicator (Weekday 6-8 AM, 3-5 PM)
        df_feat["IsRushHour"] = ((df_feat["IsWeekend"] == 0) & df_feat["Hour"].isin([6, 7, 8, 15, 16, 17])).astype(int)
        
        # Weather Impact Score
        df_feat["WeatherImpactScore"] = (df_feat["rain_1h"] * 2.0) + (df_feat["snow_1h"] * 10.0) + (df_feat["clouds_all"] / 100.0)
        df_feat.loc[df_feat["weather_main"].isin(["Squall", "Thunderstorm", "Snow"]), "WeatherImpactScore"] += 2.0
        df_feat.loc[df_feat["weather_main"].isin(["Rain", "Drizzle", "Mist", "Fog", "Smoke", "Haze"]), "WeatherImpactScore"] += 1.0
        df_feat.loc[df_feat["weather_main"].isin(["Clouds"]), "WeatherImpactScore"] += 0.5
        
        return df_feat
        
    X_train = add_engineered_features(X_train)
    X_test = add_engineered_features(X_test)
    print("Added features: 'IsWeekend', 'IsPeakHour', 'IsRushHour', 'WeatherImpactScore', 'traffic_density_score'.")
    
    # Save the categories list for UI drop downs
    unique_holidays = sorted(X_train["holiday"].unique().tolist())
    unique_weather_main = sorted(X_train["weather_main"].unique().tolist())
    unique_weather_desc = sorted(X_train["weather_description"].unique().tolist())
    
    # 6. Categorical Encoding (One-Hot Encoding)
    cat_cols = ["holiday", "weather_main", "weather_description"]
    
    X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    
    # Align to ensure identical column ordering and drop/fill mismatched columns
    # We drop date_time prior to encoding alignment
    X_train_encoded.drop(columns=["date_time"], inplace=True)
    X_test_encoded.drop(columns=["date_time"], inplace=True)
    
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join="left", axis=1, fill_value=0)
    
    # Convert dummy columns to integers (0 or 1)
    dummy_cols = [c for c in X_train_encoded.columns if any(cat in c for cat in cat_cols)]
    X_train_encoded[dummy_cols] = X_train_encoded[dummy_cols].astype(int)
    X_test_encoded[dummy_cols] = X_test_encoded[dummy_cols].astype(int)
    
    # 7. Scaling Numerical Columns
    num_cols = ["temp", "rain_1h", "snow_1h", "clouds_all", "Year", "Month", "Day", "Hour", "DayOfWeek", "WeatherImpactScore", "traffic_density_score"]
    
    scaler = StandardScaler()
    X_train_scaled = X_train_encoded.copy()
    X_test_scaled = X_test_encoded.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train_encoded[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test_encoded[num_cols])
    
    print("Standardized numerical features.")
    
    # Save preprocessing objects for Streamlit deployment
    preprocessing_artifacts = {
        "scaler": scaler,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "encoded_cols_order": X_train_encoded.columns.tolist(),
        "density_mapping": density_mapping,
        "train_mean_fallback": train_mean_fallback,
        "unique_holidays": unique_holidays,
        "unique_weather_main": unique_weather_main,
        "unique_weather_desc": unique_weather_desc,
        "temp_median": valid_temp_median
    }
    joblib.dump(preprocessing_artifacts, "preprocessing_artifacts.pkl", compress=3)
    print("Saved preprocessing artifacts to preprocessing_artifacts.pkl")
    
    print("\n--- Phase 2: Model Training & Tuning ---")
    
    # 1. Random Forest Regressor
    print("Training Random Forest Regressor (this may take a minute)...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    print("Random Forest Regressor training completed.")
    
    # 2. Hyperparameter Tuning using Optuna for LightGBM
    print("Starting Optuna Tuning for LightGBM...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def lgb_objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "n_estimators": 150,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1
        }
        
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        rmse_scores = []
        for train_idx, val_idx in cv.split(X_train_scaled, y_train):
            X_tr, y_tr = X_train_scaled.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train_scaled.iloc[val_idx], y_train.iloc[val_idx]
            
            clf = lgb.LGBMRegressor(**params)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_va)
            rmse_scores.append(root_mean_squared_error(y_va, preds))
            
        return np.mean(rmse_scores)
        
    lgb_study = optuna.create_study(direction="minimize")
    lgb_study.optimize(lgb_objective, n_trials=10)
    print(f"LightGBM best CV RMSE: {lgb_study.best_value:.4f}")
    print(f"LightGBM best params: {lgb_study.best_params}")
    
    best_lgb_params = lgb_study.best_params
    best_lgb_params["n_estimators"] = 300
    best_lgb_params["random_state"] = 42
    best_lgb_params["verbose"] = -1
    best_lgb_params["n_jobs"] = -1
    
    print("Training Best LightGBM Regressor...")
    best_lgb = lgb.LGBMRegressor(**best_lgb_params)
    best_lgb.fit(X_train_scaled, y_train)
    joblib.dump(best_lgb, "best_lgb_model.pkl", compress=3)
    
    # 3. Hyperparameter Tuning using Optuna for XGBoost
    print("Starting Optuna Tuning for XGBoost...")
    def xgb_objective(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "n_estimators": 150,
            "random_state": 42,
            "n_jobs": -1
        }
        
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        rmse_scores = []
        for train_idx, val_idx in cv.split(X_train_scaled, y_train):
            X_tr, y_tr = X_train_scaled.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train_scaled.iloc[val_idx], y_train.iloc[val_idx]
            
            clf = xgb.XGBRegressor(**params)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_va)
            rmse_scores.append(root_mean_squared_error(y_va, preds))
            
        return np.mean(rmse_scores)
        
    xgb_study = optuna.create_study(direction="minimize")
    xgb_study.optimize(xgb_objective, n_trials=10)
    print(f"XGBoost best CV RMSE: {xgb_study.best_value:.4f}")
    print(f"XGBoost best params: {xgb_study.best_params}")
    
    best_xgb_params = xgb_study.best_params
    best_xgb_params["n_estimators"] = 300
    best_xgb_params["random_state"] = 42
    best_xgb_params["n_jobs"] = -1
    
    print("Training Best XGBoost Regressor...")
    best_xgb = xgb.XGBRegressor(**best_xgb_params)
    best_xgb.fit(X_train_scaled, y_train)
    joblib.dump(best_xgb, "best_xgb_model.pkl", compress=3)
    
    # 4. Ensemble Model (55% LightGBM + 45% XGBoost)
    print("Building Ensemble Model...")
    ensemble = TrafficEnsembleRegressor(best_lgb, best_xgb, lgb_weight=0.55, xgb_weight=0.45)
    joblib.dump(ensemble, "best_ensemble_model.pkl", compress=3)
    
    # Save RF too
    joblib.dump(rf, "best_rf_model.pkl", compress=3)
    
    print("\n--- Phase 3: Model Evaluation ---")
    
    models = {
        "Random Forest": rf,
        "XGBoost (Tuned)": best_xgb,
        "LightGBM (Tuned)": best_lgb,
        "Ensemble (55% LGB + 45% XGB)": ensemble
    }
    
    results_summary = []
    
    for name, model in models.items():
        preds = model.predict(X_test_scaled)
        
        # Calculate Metrics
        r2 = r2_score(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mape = mean_absolute_percentage_error(y_test, preds)
        
        results_summary.append({
            "Model": name,
            "R² Score": r2,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape
        })
        
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv("model_evaluation_metrics.csv", index=False)
    print("\nModel Evaluation Performance Metrics Table:")
    print(results_df.to_string(index=False))
    
    print("\n--- Phase 4: Visualizations ---")
    
    # 1. Traffic Demand Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(y, kde=True, bins=50, color="#2B6CB0")
    plt.title("Traffic Volume (Demand) Distribution", fontsize=14, pad=15)
    plt.xlabel("Traffic Volume (Vehicles/Hour)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig("traffic_demand_distribution.png", dpi=300)
    plt.close()
    print("Saved traffic_demand_distribution.png")
    
    # 2. Hourly Traffic Analysis Plot (Weekdays vs Weekends)
    plt.figure(figsize=(10, 6))
    # Aggregate data for clean line plot
    agg_df = df.groupby(["IsWeekend", "Hour"])["traffic_volume"].mean().reset_index()
    sns.lineplot(data=agg_df, x="Hour", y="traffic_volume", hue="IsWeekend", palette=["#2B6CB0", "#E53E3E"], linewidth=2.5, marker="o")
    plt.title("Hourly Traffic Volume Analysis (Weekday vs Weekend)", fontsize=14, pad=15)
    plt.xlabel("Hour of the Day (0-23)", fontsize=12)
    plt.ylabel("Average Traffic Volume", fontsize=12)
    plt.legend(["Weekday", "Weekend"], loc="upper left")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig("hourly_traffic_analysis.png", dpi=300)
    plt.close()
    print("Saved hourly_traffic_analysis.png")
    
    # 3. Weather Impact Analysis Plot
    plt.figure(figsize=(12, 6))
    weather_order = df.groupby("weather_main")["traffic_volume"].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x="weather_main", y="traffic_volume", order=weather_order, palette="viridis")
    plt.title("Traffic Volume Distribution by Weather Condition", fontsize=14, pad=15)
    plt.xlabel("Weather Condition", fontsize=12)
    plt.ylabel("Traffic Volume (Vehicles/Hour)", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("weather_impact_analysis.png", dpi=300)
    plt.close()
    print("Saved weather_impact_analysis.png")
    
    # 4. Correlation Heatmap Plot
    plt.figure(figsize=(10, 8))
    # Select numerical columns including engineered ones
    num_df = X_train.copy()
    num_df["traffic_volume"] = y_train
    corr_cols = ["temp", "rain_1h", "snow_1h", "clouds_all", "Hour", "DayOfWeek", "IsWeekend", "WeatherImpactScore", "traffic_density_score", "traffic_volume"]
    corr_matrix = num_df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title("Correlation Matrix of Traffic and Environment Features", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=300)
    plt.close()
    print("Saved correlation_heatmap.png")
    
    # 5. Feature Importance Plot (LightGBM)
    plt.figure(figsize=(12, 8))
    importance_df = pd.DataFrame({
        "Feature": X_train_scaled.columns,
        "Importance": best_lgb.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    # Keep top 15 features
    sns.barplot(data=importance_df.head(15), x="Importance", y="Feature", palette="viridis")
    plt.title("Top 15 Feature Importances (LightGBM Tuned)", fontsize=14, pad=15)
    plt.xlabel("Feature Importance Score", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    plt.close()
    print("Saved feature_importance.png")
    
    # 6. Actual vs Predicted Plot (Ensemble Model on Test Set)
    plt.figure(figsize=(10, 6))
    ensemble_preds = ensemble.predict(X_test_scaled)
    # Take a sample of 1000 points for clear visualization
    sample_indices = np.random.choice(len(y_test), size=min(1000, len(y_test)), replace=False)
    sns.scatterplot(x=y_test.iloc[sample_indices], y=ensemble_preds[sample_indices], alpha=0.5, color="#2B6CB0")
    plt.plot([0, 7500], [0, 7500], "r--", linewidth=2, label="Perfect Fit Line")
    plt.title("Actual vs Predicted Traffic Demand (Ensemble Model)", fontsize=14, pad=15)
    plt.xlabel("Actual Traffic Volume (Vehicles/Hour)", fontsize=12)
    plt.ylabel("Predicted Traffic Volume (Vehicles/Hour)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=300)
    plt.close()
    print("Saved actual_vs_predicted.png")
    
    print("\n--- Pipeline finished successfully! Models and plots saved. ---")

if __name__ == "__main__":
    main()
