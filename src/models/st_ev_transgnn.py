import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple

# --- Graph Convolutional Layer (GCN) ---

class GraphConvolution(nn.Module):
    """
    Simple Graph Convolutional Layer.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        """
        Input: (B, N, F_in) where B is batch size, N is number of nodes, F_in is input features
        Adj: (N, N) where N is number of nodes
        Output: (B, N, F_out)
        """
        support = torch.matmul(input, self.weight) # (B, N, F_in) * (F_in, F_out) -> (B, N, F_out)
        output = torch.matmul(adj, support) # (N, N) * (B, N, F_out) -> (B, N, F_out)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

# --- Spatio-Temporal EV Transformer-GNN (ST-EV-TransGNN) ---

class ST_EV_TransGNN(nn.Module):
    """
    Hybrid Spatio-Temporal Transformer-GNN Model for EV Demand Forecasting.
    
    The model processes the time series data with a Transformer Encoder for temporal
    dependencies and then applies a GCN layer for spatial dependencies.
    """
    def __init__(self, 
                 num_features: int, 
                 num_stations: int, 
                 seq_len: int, 
                 pred_len: int, 
                 d_model: int = 64, 
                 nhead: int = 4, 
                 num_encoder_layers: int = 2, 
                 dim_feedforward: int = 128, 
                 dropout: float = 0.1):
        super(ST_EV_TransGNN, self).__init__()
        
        self.num_stations = num_stations
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # 1. Feature Projection/Embedding Layer
        # Input is (B, L, F), we want to project F to d_model
        self.input_projection = nn.Linear(num_features, d_model)
        
        # 2. Transformer Encoder for Temporal Feature Extraction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # 3. Graph Convolutional Layer for Spatial Feature Extraction
        # The GCN operates on the last time step's output from the Transformer
        # We assume the output of the Transformer is (B, L, d_model)
        # We will take the last time step (B, d_model) and reshape it to (B, N, F_in)
        # where N is num_stations and F_in is d_model / num_stations
        
        # The input to GCN will be (B, N, d_model)
        self.gcn = GraphConvolution(d_model, d_model)
        
        # 4. Forecasting Head
        # Input to the head is (B, N * d_model) after flattening the GCN output
        self.fc_out = nn.Sequential(
            nn.Linear(num_stations * d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, num_stations * pred_len)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ST-EV-TransGNN model.
        
        Args:
            x: Input tensor (B, L, F) - Batch, Sequence Length, Features
            adj: Adjacency matrix (N, N) - Number of Stations, Number of Stations
            
        Returns:
            Output tensor (B, P) - Batch, Prediction Length (Total Aggregated Demand)
        """
        B, L, F = x.shape
        
        # 1. Feature Projection
        x = self.input_projection(x) # (B, L, F) -> (B, L, d_model)
        
        # 2. Temporal Feature Extraction (Transformer)
        # Transformer expects (B, L, d_model)
        temporal_features = self.transformer_encoder(x) # (B, L, d_model)
        
        # 3. Spatial Feature Extraction (GCN)
        # We take the last time step's output for forecasting
        # (B, L, d_model) -> (B, d_model)
        last_step_features = temporal_features[:, -1, :] 
        
        # The GCN is designed to model spatial correlation between stations.
        # Since the input 'x' is aggregated total demand features, we must simplify 
        # the GCN application to work with the temporal feature vector (B, d_model).
        # We assume the d_model features are a representation of the N stations.
        
        # Replicate the feature vector N times: (B, d_model) -> (B, N, d_model)
        # This is a strong simplification, but necessary for the current aggregated input.
        last_step_features_reshaped = last_step_features.unsqueeze(1).repeat(1, self.num_stations, 1)
        
        # Apply GCN: (B, N, d_model) -> (B, N, d_model)
        spatial_features = self.gcn(last_step_features_reshaped, adj)
        
        # Flatten for FC layer: (B, N, d_model) -> (B, N * d_model)
        flattened_features = spatial_features.reshape(B, -1)
            
        # 4. Forecasting Head
        # Output is (B, N * P)
        output = self.fc_out(flattened_features)
        
        # Reshape to (B, N, P)
        output = output.reshape(B, self.num_stations, self.pred_len)
        
        # For the current project (aggregated demand), we only need the total demand.
        # We will sum the predictions across all stations for the final output.
        # (B, N, P) -> (B, P)
        final_output = output.sum(dim=1)
        
        return final_output

# --- Data Preparation Utility ---

def create_sequences(data: pd.DataFrame, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates input sequences (X) and target sequences (Y) for the model.
    
    Args:
        data: DataFrame with all features, including the target 'ev_demand_kW'.
        seq_len: Length of the input sequence (look-back window).
        pred_len: Length of the prediction horizon.
        
    Returns:
        Tuple of (X, Y) as numpy arrays.
    """
    X, Y = [], []
    target_col_index = data.columns.get_loc("ev_demand_kW")
    
    data_values = data.values
    
    for i in range(len(data) - seq_len - pred_len + 1):
        # Input sequence X: features from t to t + seq_len - 1
        x_seq = data_values[i : i + seq_len, :]
        
        # Target sequence Y: target variable from t + seq_len to t + seq_len + pred_len - 1
        # We only predict the 'ev_demand_kW'
        y_seq = data_values[i + seq_len : i + seq_len + pred_len, target_col_index]
        
        X.append(x_seq)
        Y.append(y_seq)
        
    return np.array(X), np.array(Y)

def create_adjacency_matrix(num_stations: int) -> torch.Tensor:
    """
    Creates a simple, fully connected adjacency matrix for the GCN.
    In a real project, this would be based on geographical distance or grid topology.
    """
    # Fully connected graph with self-loops
    adj = np.ones((num_stations, num_stations))
    np.fill_diagonal(adj, 1) # Self-loops are already included
    
    # Simple normalization (D^-1 * A)
    D = np.diag(np.sum(adj, axis=1))
    D_inv = np.linalg.inv(D)
    adj_norm = np.dot(D_inv, adj)
    
    return torch.from_numpy(adj_norm).float()

# --- Main Training Script (for next phase) ---

if __name__ == "__main__":
    # Example usage for testing the model structure
    
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "full_forecasting_data.csv")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # 2. Parameters
    SEQ_LEN = 24 # Look-back window (24 hours)
    PRED_LEN = 4 # Prediction horizon (4 hours)
    NUM_FEATURES = df.shape[1]
    NUM_STATIONS = 5 # Based on synthetic data generation
    
    # 3. Create Sequences
    X_np, Y_np = create_sequences(df, SEQ_LEN, PRED_LEN)
    
    # 4. Create Adjacency Matrix
    adj_matrix = create_adjacency_matrix(NUM_STATIONS)
    
    # 5. Initialize Model
    model = ST_EV_TransGNN(
        num_features=NUM_FEATURES, 
        num_stations=NUM_STATIONS, 
        seq_len=SEQ_LEN, 
        pred_len=PRED_LEN
    )
    
    # 6. Test Forward Pass
    X_tensor = torch.from_numpy(X_np).float()
    
    # Take a small batch for testing
    test_batch_X = X_tensor[:16]
    
    print(f"Input X shape: {test_batch_X.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(test_batch_X, adj_matrix)
        
    print(f"Output shape (B, P): {output.shape}")
    print(f"Expected output shape (Batch, Pred_Len): ({test_batch_X.shape[0]}, {PRED_LEN})")
    
    assert output.shape == (test_batch_X.shape[0], PRED_LEN)
    print("\nModel structure test passed successfully.")
