import torch
import numpy as np
import pandas as pd
import os
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import plotly.express as px
import sys
# Add the project root to the system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.st_ev_transgnn import ST_EV_TransGNN, create_sequences, create_adjacency_matrix
from src.train import SEQ_LEN, PRED_LEN, NUM_STATIONS, TRAIN_SPLIT, MODEL_PATH, FULL_DATA_PATH

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Evaluation Function ---

def evaluate_model():
    # 1. Load Data and Model
    print("Loading data, model, and scaler...")
    df = pd.read_csv(FULL_DATA_PATH, index_col=0, parse_dates=True)
    scaler = load(SCALER_PATH)
    
    X_np, Y_np = create_sequences(df, SEQ_LEN, PRED_LEN)
    
    # Split data (same as training)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_np, Y_np, test_size=1 - TRAIN_SPLIT, shuffle=False
    )
    
    # Convert to PyTorch tensors
    X_test_tensor = torch.from_numpy(X_test).float()
    
    NUM_FEATURES = X_test.shape[2]
    adj_matrix = create_adjacency_matrix(NUM_STATIONS)
    
    model = ST_EV_TransGNN(
        num_features=NUM_FEATURES, 
        num_stations=NUM_STATIONS, 
        seq_len=SEQ_LEN, 
        pred_len=PRED_LEN
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # 2. Generate Predictions
    print("Generating predictions...")
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor, adj_matrix).numpy()
    
    # 3. Inverse Transform Predictions and Actuals
    # The target variable is 'ev_demand_kW' which is the first column in the scaled data.
    # We need to create a dummy array with the correct shape for inverse transform.
    
    # Get the index of the target column
    target_col_index = df.columns.get_loc("ev_demand_kW")
    
    # Inverse transform function
    def inverse_transform_target(scaled_data, target_index, scaler, num_features):
        # Create a zero array with the shape (num_samples, num_features)
        dummy = np.zeros((scaled_data.shape[0], num_features))
        # Place the scaled target data into the correct column
        dummy[:, target_index] = scaled_data
        # Inverse transform the dummy array
        inverse_transformed = scaler.inverse_transform(dummy)
        # Return only the target column
        return inverse_transformed[:, target_index]

    # Reshape Y_test and predictions_scaled to (num_samples * pred_len, 1)
    Y_test_flat = Y_test.flatten()
    predictions_flat = predictions_scaled.flatten()
    
    # Get the number of features from the original DataFrame
    num_features = df.shape[1]
    
    # Inverse transform
    actuals = inverse_transform_target(Y_test_flat, target_col_index, scaler, num_features)
    predictions = inverse_transform_target(predictions_flat, target_col_index, scaler, num_features)
    
    # 4. Numerical Evaluation (Results & Analysis - 20%)
    print("\n--- Numerical Evaluation ---")
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    
    metrics = {
        "Mean Squared Error (MSE)": mse,
        "Root Mean Squared Error (RMSE)": rmse,
        "Mean Absolute Error (MAE)": mae
    }
    
    metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    print(metrics_df.to_markdown(index=False))
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "performance_metrics.csv"), index=False)
    
    # 5. Visual Evaluation (Results & Analysis - 20%)
    print("\n--- Visual Evaluation ---")
    
    # Reshape back to (num_samples, pred_len) for time-based plotting
    actuals_reshaped = actuals.reshape(-1, PRED_LEN)
    predictions_reshaped = predictions.reshape(-1, PRED_LEN)
    
    # Take the first prediction step (t+1) for a simple time series plot
    actuals_t1 = actuals_reshaped[:, 0]
    predictions_t1 = predictions_reshaped[:, 0]
    
    # Create a time index for the test set
    # The time index for the test set starts at the first prediction point.
    # The total number of samples in the processed data is len(df).
    # The training set is the first TRAIN_SPLIT of the sequences created.
    # The sequences start at index SEQ_LEN of the original df.
    
    # Index of the first sample in the test set (in the original df index)
    test_start_idx_in_df = len(df) - len(Y_np) + len(Y_train)
    
    # The time index for the actuals/predictions is the time of the first prediction (t+1)
    # which is at index test_start_idx_in_df + SEQ_LEN
    
    # The length of the test set is len(actuals_t1)
    num_test_samples = len(actuals_t1)
    
    # The correct start index in the original df.index is:
    # (Length of training sequences) + (Sequence Length)
    # The sequences start at df.index[SEQ_LEN]
    # The test sequences start at df.index[SEQ_LEN + len(Y_train)]
    
    # Let's use the index of the last training sequence + 1
    last_train_seq_end_index = len(Y_train) + SEQ_LEN
    
    # The first prediction is for the time step after the last input of the first test sequence.
    # The first test sequence starts at index len(Y_train)
    # The first prediction is at index len(Y_train) + SEQ_LEN
    
    test_time_index_start = df.index[len(Y_train) + SEQ_LEN]
    test_time_index_end = df.index[len(Y_train) + SEQ_LEN + num_test_samples - 1]
    
    # Re-slicing the index to match the length of the test set
    test_time_index = df.index[len(Y_train) + SEQ_LEN : len(Y_train) + SEQ_LEN + num_test_samples]
    
    plot_df = pd.DataFrame({
        "Time": test_time_index,
        "Actual Demand (kW)": actuals_t1,
        "Predicted Demand (kW)": predictions_t1
    })
    
    # Plot 1: Time Series Plot (Matplotlib for static report)
    plt.figure(figsize=(15, 6))
    plt.plot(plot_df["Time"], plot_df["Actual Demand (kW)"], label="Actual Demand", color="blue", alpha=0.7)
    plt.plot(plot_df["Time"], plot_df["Predicted Demand (kW)"], label="Predicted Demand", color="red", linestyle="--", alpha=0.7)
    plt.title("EV Charging Demand Forecast (t+1 hour)")
    plt.xlabel("Time")
    plt.ylabel("Demand (kW)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "forecast_time_series.png"))
    print(f"Saved static plot to {RESULTS_DIR}/forecast_time_series.png")
    
    # Plot 2: Scatter Plot (Matplotlib for static report)
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals_t1, predictions_t1, alpha=0.5)
    max_val = max(actuals_t1.max(), predictions_t1.max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.title("Actual vs. Predicted Demand (t+1 hour)")
    plt.xlabel("Actual Demand (kW)")
    plt.ylabel("Predicted Demand (kW)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "forecast_scatter.png"))
    print(f"Saved static plot to {RESULTS_DIR}/forecast_scatter.png")
    
    # Plot 3: Interactive Plot (Plotly for UI/Application)
    fig = px.line(plot_df, x="Time", y=["Actual Demand (kW)", "Predicted Demand (kW)"], 
                  title="Interactive EV Charging Demand Forecast (t+1 hour)")
    fig.write_html(os.path.join(RESULTS_DIR, "forecast_interactive.html"))
    print(f"Saved interactive plot to {RESULTS_DIR}/forecast_interactive.html")
    
    print("\nEvaluation complete.")

if __name__ == "__main__":
    evaluate_model()
