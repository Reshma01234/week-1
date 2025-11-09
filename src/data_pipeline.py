import pandas as pd
import numpy as np
# from acnportal import acndata # API client requires token
import requests
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, List
import os

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
EV_DATA_PATH = os.path.join(DATA_DIR, "ev_demand.csv")
FULL_DATA_PATH = os.path.join(DATA_DIR, "full_forecasting_data.csv")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")

# --- Data Ingestion and Aggregation ---

def fetch_and_aggregate_ev_data() -> pd.DataFrame:
    """
    Generates a synthetic, multi-station EV charging demand time series.
    This simulates the spatio-temporal nature required for the GNN model.
    """
    print("Generating synthetic multi-station EV demand data...")
    
    # Parameters for synthetic data
    start_date = "2024-01-01"
    end_date = "2024-07-01"
    freq = "H"
    num_stations = 5
    
    time_index = pd.date_range(start=start_date, end=end_date, freq=freq, inclusive="left")
    df = pd.DataFrame(index=time_index)
    
    # Base demand pattern (peaks in morning and evening)
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    
    # Daily pattern: higher demand during work hours (8-10) and evening (17-20)
    daily_pattern = np.where((df["hour"] >= 8) & (df["hour"] <= 10), 1, 0) + \
                    np.where((df["hour"] >= 17) & (df["hour"] <= 20), 1.5, 0)
    
    # Weekly pattern: lower demand on weekends (day_of_week 5=Sat, 6=Sun)
    weekly_pattern = np.where(df["day_of_week"].isin([5, 6]), 0.5, 1)
    
    # Trend and Noise
    trend = np.linspace(0, 0.5, len(df))
    
    # Generate demand for multiple stations
    station_data = {}
    for i in range(num_stations):
        # Introduce spatial correlation and station-specific bias
        # Station 0 is the "main" station, others are correlated but with noise/offset
        base_demand = 50 * daily_pattern * weekly_pattern * (1 + trend)
        
        # Station-specific multiplier and noise
        multiplier = 1 + (i * 0.1) # Stations get progressively busier
        noise = np.random.normal(0, 5 + i * 2, len(df))
        
        demand = base_demand * multiplier + noise
        demand = np.clip(demand, 0, None) # Demand cannot be negative
        
        station_id = f"station_{i+1}"
        station_data[station_id] = demand
        
    # Combine into a single DataFrame with MultiIndex (Time, Station)
    multi_index = pd.MultiIndex.from_product([time_index, [f"station_{i+1}" for i in range(num_stations)]], names=["connectionTime", "station_id"])
    
    # Flatten the station data into a single series
    ev_demand_series = pd.concat([pd.Series(v, index=time_index) for v in station_data.values()], keys=station_data.keys()).swaplevel(0, 1).sort_index()
    ev_demand_series.name = "ev_demand_kW"
    
    # Convert the series back to a DataFrame with a single time index for the next step
    # For the GNN part, we will reshape this data later.
    # For now, we aggregate the total demand across all stations for the main time series.
    total_hourly_demand = ev_demand_series.groupby(level=0).sum().to_frame()
    
    print(f"Synthetic multi-station EV data generated. Total hours: {len(total_hourly_demand)}")
    
    # Save the multi-station data for later use in GNN feature creation
    multi_station_df = ev_demand_series.reset_index()
    multi_station_df.to_csv(os.path.join(DATA_DIR, "multi_station_demand.csv"), index=False)
    
    return total_hourly_demand

# --- Synthetic Auxiliary Data Generation ---

def generate_synthetic_auxiliary_data(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Generates synthetic, yet realistic, data for Solar, Wind, Weather, and Grid features.
    This simulates the RARE Power, NOAA HRRR, and NREL Grid datasets.
    """
    print("Generating synthetic auxiliary data...")
    
    df = pd.DataFrame(index=index)
    
    # 1. Weather Data (Simulating NOAA HRRR)
    # Temperature (sinusoidal with daily and annual cycle, plus noise)
    df["hour"] = df.index.hour
    df["day_of_year"] = df.index.dayofyear
    df["temp_C"] = 15 + 10 * np.sin(2 * np.pi * (df["day_of_year"] - 80) / 365) + \
                   5 * np.sin(2 * np.pi * (df["hour"] - 12) / 24) + np.random.normal(0, 2, len(df))
    
    # Cloud Cover (higher during the day, random noise)
    df["cloud_cover"] = np.clip(0.5 + 0.3 * np.sin(2 * np.pi * (df["hour"] - 8) / 24) + np.random.normal(0, 0.1, len(df)), 0, 1)
    
    # 2. Renewable Energy Data (Simulating RARE Power)
    # Solar Power (peaks around noon, zero at night, inversely related to cloud cover)
    solar_potential = np.clip(np.sin(2 * np.pi * (df["hour"] - 8) / 24), 0, 1)
    df["solar_power_kW"] = 500 * solar_potential * (1 - 0.5 * df["cloud_cover"]) + np.random.normal(0, 10, len(df))
    df["solar_power_kW"] = df["solar_power_kW"].apply(lambda x: max(0, x))
    
    # Wind Power (more random, slightly higher at night/early morning)
    df["wind_speed_m_s"] = 5 + 3 * np.sin(2 * np.pi * (df["hour"] - 4) / 24) + np.random.normal(0, 2, len(df))
    df["wind_power_kW"] = 200 * np.clip(df["wind_speed_m_s"] / 10, 0, 1) ** 3 + np.random.normal(0, 5, len(df))
    df["wind_power_kW"] = df["wind_power_kW"].apply(lambda x: max(0, x))
    
    # 3. Grid Stability Feature (Simulating NREL Grid Datasets)
    # Grid Frequency Deviation (proxy for stability, higher during peak demand hours)
    df["grid_freq_dev"] = 0.01 * np.clip(np.sin(2 * np.pi * (df["hour"] - 17) / 24), 0, 1) + np.random.normal(0, 0.001, len(df))
    
    df = df.drop(columns=["hour", "day_of_year"])
    print("Synthetic auxiliary data generated.")
    return df

# --- Feature Engineering and Preprocessing ---

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates temporal and lagged features for time series forecasting.
    """
    print("Creating temporal features...")
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek # Monday=0, Sunday=6
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    
    # Cyclical features for time
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    # Lagged EV Demand (Target Variable)
    print("Creating lagged features...")
    for lag in [1, 2, 3, 24, 48, 168]: # Last 1, 2, 3 hours, last day, last 2 days, last week
        df[f"lag_{lag}_ev_demand"] = df["ev_demand_kW"].shift(lag)
        
    # Lagged Renewable/Weather Features (Contextual Features)
    for col in ["solar_power_kW", "wind_power_kW", "temp_C"]:
        df[f"lag_1_{col}"] = df[col].shift(1)
        
    # Drop rows with NaN values created by lagging
    df = df.dropna()
    
    print(f"Feature engineering complete. Total samples: {len(df)}")
    return df

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scales the data and saves the scaler.
    """
    from joblib import dump
    
    print("Scaling data...")
    # We will scale all numerical features for optimal neural network performance.
    df = df.select_dtypes(include=[np.number]) # Select only numerical columns
    
    # Initialize and fit scaler on all numerical features
    scaler = MinMaxScaler()
    df_scaled_values = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled_values, index=df.index, columns=df.columns)
    
    # Save the scaler for inverse transformation later
    dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")
    
    return df_scaled, scaler

def run_pipeline() -> pd.DataFrame:
    """
    Runs the full data pipeline.
    """
    # 1. Fetch and Aggregate EV Data
    ev_df = fetch_and_aggregate_ev_data()
    if ev_df.empty:
        return pd.DataFrame()

    # 2. Generate Auxiliary Data
    aux_df = generate_synthetic_auxiliary_data(ev_df.index)
    
    # 3. Merge DataFrames
    # The synthetic EV data is already aligned with the time index
    full_df = ev_df.join(aux_df, how="inner")
    
    # 4. Feature Engineering
    featured_df = create_features(full_df)
    
    # 5. Preprocessing (Scaling)
    df_scaled, _ = preprocess_data(featured_df)
    
    # 6. Save final processed data
    df_scaled.to_csv(FULL_DATA_PATH)
    print(f"Final processed data saved to {FULL_DATA_PATH}")
    
    return df_scaled

if __name__ == "__main__":
    # To run the script and see the output
    processed_data = run_pipeline()
    if not processed_data.empty:
        print("\nProcessed Data Head:")
        print(processed_data.head())
        print("\nProcessed Data Columns:")
        print(processed_data.columns.tolist())
        print("\nData Statistics:")
        print(processed_data.describe())
