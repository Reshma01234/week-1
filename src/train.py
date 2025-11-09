import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import sys
import os
# Add the project root to the system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from joblib import load
from sklearn.model_selection import train_test_split
from src.models.st_ev_transgnn import ST_EV_TransGNN, create_sequences, create_adjacency_matrix

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

FULL_DATA_PATH = os.path.join(DATA_DIR, "full_forecasting_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "st_ev_transgnn_model.pth")

# --- Hyperparameters ---
SEQ_LEN = 24  # Look-back window (hours)
PRED_LEN = 4  # Prediction horizon (hours)
NUM_STATIONS = 5 # From synthetic data generation
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50 # Reduced for sandbox execution speed
TRAIN_SPLIT = 0.8

# --- Training Function ---

def train_model():
    # 1. Load and Prepare Data
    print("Loading and preparing data...")
    df = pd.read_csv(FULL_DATA_PATH, index_col=0, parse_dates=True)
    
    X_np, Y_np = create_sequences(df, SEQ_LEN, PRED_LEN)
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_np, Y_np, test_size=1 - TRAIN_SPLIT, shuffle=False
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    Y_train_tensor = torch.from_numpy(Y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    Y_test_tensor = torch.from_numpy(Y_test).float()
    
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model and Components
    NUM_FEATURES = X_train.shape[2]
    adj_matrix = create_adjacency_matrix(NUM_STATIONS)
    
    model = ST_EV_TransGNN(
        num_features=NUM_FEATURES, 
        num_stations=NUM_STATIONS, 
        seq_len=SEQ_LEN, 
        pred_len=PRED_LEN
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            
            # The model is designed to output (B, P) which is the sum of all station predictions
            # The target Y is also the sum of all station demands (B, P)
            output = model(batch_X, adj_matrix)
            
            loss = criterion(output, batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 4. Evaluation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                output = model(batch_X, adj_matrix)
                loss = criterion(output, batch_Y)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")

    # 5. Save Model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
