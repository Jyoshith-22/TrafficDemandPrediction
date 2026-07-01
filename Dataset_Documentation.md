# Dataset Documentation: Metro Interstate Traffic Volume

## 1. Dataset Overview
This dataset contains hourly traffic volume on Interstate 94 (I-94) Eastbound at a specific sensor location in Minneapolis-St. Paul, Minnesota, coupled with hourly weather features and holiday markers from 2012 to 2018.

* **Source Location:** I-94 Eastbound, Minneapolis-St. Paul, MN
* **Total Records:** 48,204 rows
* **Total Columns:** 9

## 2. Features Description

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `holiday` | Categorical (String) | Indicates whether the day is a US National Holiday (e.g. Columbus Day, Veterans Day, Labor Day, Thanksgiving Day, etc.) or a normal day. |
| `temp` | Continuous (Float) | Average hourly temperature in Kelvin (K). |
| `rain_1h` | Continuous (Float) | Hourly rainfall volume recorded in millimeters (mm). |
| `snow_1h` | Continuous (Float) | Hourly snowfall volume recorded in millimeters (mm). |
| `clouds_all` | Integer (Int) | Cloudiness percentage from 0% to 100%. |
| `weather_main` | Categorical (String) | Short categorical weather descriptor (e.g. Clear, Clouds, Rain, Snow, Drizzle, Mist, Fog, Haze, Thunderstorm, Squall, Smoke). |
| `weather_description` | Categorical (String) | Detailed weather descriptor (e.g. scattered clouds, light rain, heavy intensity rain, proximity thunderstorm, etc.). |
| `date_time` | DateTime (String) | Hourly timestamp in the format `YYYY-MM-DD HH:MM:SS`. |
| `traffic_volume` | Integer (Int) | **Target Variable**. The hourly volume of vehicles traveling through the corridor. Range: 0 to 7,280. |

## 3. Data Cleaning & Handling Anomalies
1. **Holiday Missing Values:**
   - The raw `holiday` column contains 48,143 null values representing standard non-holiday days. We impute these nulls with `"None"` to convert it into a complete categorical feature.
2. **Temperature Sensor Errors (0 Kelvin):**
   - Identified 10 records with a temperature of `0.0` Kelvin (-273.15 °C). Since these values are physically impossible and would distort the model's feature scaling, they were imputed with the dataset's median temperature of `288.28` Kelvin.
3. **Extreme Precipitation Outlier:**
   - Identified a single record containing a rainfall volume of `9,831.3` mm/hour (nearly 10 meters of rain in one hour). This anomalous value is a recording error and was deleted from the dataset to prevent biasing the models.

## 4. Preprocessing and Feature Engineering
1. **Temporal Features:**
   - Decomposed the `date_time` string into `Year`, `Month`, `Day`, `Hour`, and `DayOfWeek`.
2. **Weekend Flag:**
   - `IsWeekend` = 1 if `DayOfWeek` in `[5, 6]` (Saturday, Sunday) else 0.
3. **Peak & Rush Hour Flags:**
   - Weekdays display distinct high commuter demand.
   - `IsPeakHour` = 1 if weekday and `Hour` is 7 AM or 4 PM.
   - `IsRushHour` = 1 if weekday and `Hour` falls in morning (6-8 AM) or afternoon (3-5 PM) brackets.
4. **Weather Impact Score:**
   - A continuous custom index reflecting weather severity:
     - `WeatherImpactScore = (rain_1h * 2.0) + (snow_1h * 10.0) + (clouds_all / 100.0)`
     - Add `2.0` if `weather_main` is Squall, Thunderstorm, or Snow.
     - Add `1.0` if `weather_main` is Rain, Drizzle, Mist, Fog, Smoke, or Haze.
     - Add `0.5` if `weather_main` is Clouds.
     - Add `0.0` for Clear weather.
5. **Traffic Density Score (Target Encoding):**
   - Computes the historical average of the target variable `traffic_volume` grouped by `Hour` and `DayOfWeek` on the training dataset. This gives the models a powerful time-space traffic baseline, raising prediction R² accuracy past 97%.
6. **One-Hot Encoding:**
   - Categorical columns (`holiday`, `weather_main`, `weather_description`) were one-hot encoded, and training/serving column orders were aligned.
7. **Z-score Standardization:**
   - Numerical columns were scaled using a standard scaler to maintain scale uniformity.
