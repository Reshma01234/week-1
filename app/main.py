import os
import sys
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import json

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.st_ev_transgnn import ST_EV_TransGNN, create_sequences, create_adjacency_matrix
from src.train import SEQ_LEN, PRED_LEN, NUM_STATIONS

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

MODEL_PATH = os.path.join(MODEL_DIR, "st_ev_transgnn_model.pth")
FULL_DATA_PATH = os.path.join(DATA_DIR, "full_forecasting_data.csv")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")

# --- FastAPI Application ---

app = FastAPI(
    title="EV Charging Station Energy Demand Forecasting",
    description="Predict energy demand for EV charging stations powered by solar-wind hybrid microgrids.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# --- Global Variables for Model and Data ---

model = None
scaler = None
df = None

def load_model_and_data():
    """Load the trained model, scaler, and data."""
    global model, scaler, df
    
    try:
        # Load data
        df = pd.read_csv(FULL_DATA_PATH, index_col=0, parse_dates=True)
        
        # Load scaler
        scaler = load(SCALER_PATH)
        
        # Load model
        NUM_FEATURES = df.shape[1]
        model = ST_EV_TransGNN(
            num_features=NUM_FEATURES, 
            num_stations=NUM_STATIONS, 
            seq_len=SEQ_LEN, 
            pred_len=PRED_LEN
        )
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        
        print("Model, scaler, and data loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    """Load model and data on startup."""
    load_model_and_data()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>EV Charging Demand Forecasting</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
                max-width: 1000px;
                width: 100%;
                padding: 40px;
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
                text-align: center;
            }
            .subtitle {
                color: #666;
                text-align: center;
                margin-bottom: 30px;
                font-size: 14px;
            }
            .section {
                margin-bottom: 30px;
            }
            .section h2 {
                color: #667eea;
                font-size: 18px;
                margin-bottom: 15px;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }
            .button-group {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }
            button {
                background: #667eea;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
                transition: background 0.3s;
            }
            button:hover {
                background: #764ba2;
            }
            button.secondary {
                background: #6c757d;
            }
            button.secondary:hover {
                background: #5a6268;
            }
            .info-box {
                background: #f0f4ff;
                border-left: 4px solid #667eea;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 15px;
            }
            .info-box p {
                color: #333;
                font-size: 14px;
                line-height: 1.6;
            }
            .results {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin-top: 20px;
                display: none;
            }
            .results.show {
                display: block;
            }
            .metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .metric-card {
                background: white;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
            .metric-label {
                color: #666;
                font-size: 12px;
                text-transform: uppercase;
            }
            .metric-value {
                color: #667eea;
                font-size: 24px;
                font-weight: bold;
                margin-top: 5px;
            }
            .plot-container {
                margin-top: 20px;
                text-align: center;
            }
            .plot-container img {
                max-width: 100%;
                height: auto;
                border-radius: 5px;
            }
            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
            }
            .loading.show {
                display: block;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .error {
                background: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
                padding: 12px;
                border-radius: 5px;
                margin-top: 10px;
                display: none;
            }
            .error.show {
                display: block;
            }
            footer {
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #666;
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔋 EV Charging Station Energy Demand Forecasting</h1>
            <p class="subtitle">Predict energy demand for EV charging stations powered by solar-wind hybrid microgrids</p>
            
            <div class="section">
                <h2>📊 Features</h2>
                <div class="info-box">
                    <p>
                        <strong>State-of-the-Art Model:</strong> Spatio-Temporal Transformer-GNN (ST-EV-TransGNN)<br>
                        <strong>Prediction Horizon:</strong> 4 hours ahead<br>
                        <strong>Input Features:</strong> 25 engineered features (temporal, weather, renewable energy)<br>
                        <strong>Data:</strong> 6 months of synthetic multi-station EV charging data
                    </p>
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 Actions</h2>
                <div class="button-group">
                    <button onclick="generateForecast()">Generate Forecast</button>
                    <button class="secondary" onclick="viewResults()">View Results</button>
                    <button class="secondary" onclick="downloadReport()">Download Report</button>
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Generating forecast...</p>
            </div>
            
            <div class="error" id="error"></div>
            
            <div class="results" id="results">
                <h2>📈 Forecast Results</h2>
                <div class="metrics" id="metrics"></div>
                <div class="plot-container" id="plots"></div>
            </div>
            
            <footer>
                <p>EV Charging Station Energy Demand Forecasting for Microgrid Integration with Renewable Sources</p>
                <p>Developed for Graduation Project | 2025</p>
            </footer>
        </div>
        
        <script>
            async function generateForecast() {
                const loading = document.getElementById('loading');
                const error = document.getElementById('error');
                const results = document.getElementById('results');
                
                loading.classList.add('show');
                error.classList.remove('show');
                results.classList.remove('show');
                
                try {
                    const response = await fetch('/api/forecast');
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    const data = await response.json();
                    
                    displayResults(data);
                    results.classList.add('show');
                } catch (err) {
                    error.textContent = `Error: ${err.message}`;
                    error.classList.add('show');
                } finally {
                    loading.classList.remove('show');
                }
            }
            
            function displayResults(data) {
                const metricsDiv = document.getElementById('metrics');
                const plotsDiv = document.getElementById('plots');
                
                metricsDiv.innerHTML = '';
                plotsDiv.innerHTML = '';
                
                // Display metrics
                for (const [key, value] of Object.entries(data.metrics)) {
                    const card = document.createElement('div');
                    card.className = 'metric-card';
                    card.innerHTML = `
                        <div class="metric-label">${key}</div>
                        <div class="metric-value">${parseFloat(value).toFixed(2)}</div>
                    `;
                    metricsDiv.appendChild(card);
                }
                
                // Display plots
                const plotsHtml = `
                    <h3>Time Series Forecast (t+1 hour)</h3>
                    <img src="/static/forecast_time_series.png" alt="Time Series Plot">
                    <h3>Actual vs. Predicted (Scatter Plot)</h3>
                    <img src="/static/forecast_scatter.png" alt="Scatter Plot">
                `;
                plotsDiv.innerHTML = plotsHtml;
            }
            
            function viewResults() {
                window.open('/results/forecast_interactive.html', '_blank');
            }
            
            function downloadReport() {
                window.location.href = '/api/report';
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/api/forecast")
async def forecast():
    """Generate forecast using the trained model."""
    try:
        if model is None or df is None or scaler is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Create sequences
        X_np, Y_np = create_sequences(df, SEQ_LEN, PRED_LEN)
        
        # Use the last sequence for prediction
        X_last = X_np[-1:, :, :]  # Shape: (1, SEQ_LEN, NUM_FEATURES)
        X_tensor = torch.from_numpy(X_last).float()
        
        # Generate prediction
        adj_matrix = create_adjacency_matrix(NUM_STATIONS)
        with torch.no_grad():
            prediction_scaled = model(X_tensor, adj_matrix).numpy()
        
        # Inverse transform prediction
        target_col_index = df.columns.get_loc("ev_demand_kW")
        num_features = df.shape[1]
        
        def inverse_transform_target(scaled_data, target_index, scaler, num_features):
            dummy = np.zeros((scaled_data.shape[0], num_features))
            dummy[:, target_index] = scaled_data
            inverse_transformed = scaler.inverse_transform(dummy)
            return inverse_transformed[:, target_index]
        
        prediction = inverse_transform_target(prediction_scaled.flatten(), target_col_index, scaler, num_features)
        
        # Load metrics
        metrics_df = pd.read_csv(os.path.join(RESULTS_DIR, "performance_metrics.csv"))
        metrics = dict(zip(metrics_df["Metric"], metrics_df["Value"]))
        
        return {
            "status": "success",
            "prediction": prediction.tolist(),
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report")
async def download_report():
    """Download the performance report as a text file."""
    try:
        report_path = os.path.join(RESULTS_DIR, "performance_metrics.csv")
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="Report not found")
        
        return FileResponse(report_path, filename="performance_report.csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{file_path}")
async def get_result_file(file_path: str):
    """Serve result files (HTML, PNG, etc.)."""
    try:
        full_path = os.path.join(RESULTS_DIR, file_path)
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        if file_path.endswith(".html"):
            return FileResponse(full_path, media_type="text/html")
        elif file_path.endswith(".png"):
            return FileResponse(full_path, media_type="image/png")
        else:
            return FileResponse(full_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
