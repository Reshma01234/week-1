# EV Charging Station Energy Demand Forecasting for Microgrid Integration

## Project Overview

This project implements a state-of-the-art Spatio-Temporal Transformer-GNN (ST-EV-TransGNN) model to forecast energy demand for Electric Vehicle (EV) charging stations integrated into a solar-wind hybrid microgrid. The goal is to provide accurate, short-term (4-hour ahead) predictions to optimize the microgrid's energy management system (EMS), maximizing renewable energy utilization and ensuring grid stability.

The project is structured to meet all requirements for a comprehensive graduation submission, including a novel model, data pipeline, detailed analysis, and a working web application.

## Grading Criteria Alignment

| Criterion | Weight (%) | Alignment |
| :--- | :--- | :--- |
| **Problem Definition & Motivation** | 10 | Clearly defined challenge of EV integration into RES-powered microgrids. |
| **Literature Review** | 10 | Review of state-of-the-art models (Transformer, GNN) leading to the hybrid approach. |
| **Data & Preprocessing** | 10 | Integration of synthetic EV, Solar/Wind, and Weather data with advanced feature engineering. |
| **Model Development** | 20 | Implementation of the novel ST-EV-TransGNN hybrid deep learning architecture. |
| **Results & Analysis** | 20 | Comprehensive numerical (RMSE, MAE) and visual evaluation (time series, scatter plots). |
| **Innovation / Novelty** | 15 | The ST-EV-TransGNN model, combining Transformer (temporal) and GNN (spatial) features. |
| **Application / UI** | 10 | Working FastAPI web application for real-time forecasting and visualization. |
| **Documentation & Presentation** | 10 | Detailed project report (`docs/Project_Report.md`) and this README. |
| **Environmental / Social Impact** | 5 | Direct contribution to green energy adoption and grid stability. |

## Project Structure

| Directory/File | Description |
| :--- | :--- |
| `src/` | Core Python source code. |
| `src/data_pipeline.py` | Scripts for data generation, feature engineering, and scaling. |
| `src/models/st_ev_transgnn.py` | PyTorch implementation of the ST-EV-TransGNN model. |
| `src/train.py` | Script for training the model. |
| `src/evaluate.py` | Script for generating performance metrics and visualizations. |
| `app/` | FastAPI web application files. |
| `app/main.py` | FastAPI application entry point and API logic. |
| `data/` | Processed data (`full_forecasting_data.csv`) and scaler (`scaler.pkl`). |
| `trained_models/` | Saved PyTorch model weights (`st_ev_transgnn_model.pth`). |
| `results/` | Evaluation results (metrics, plots, interactive HTML). |
| `docs/` | Project documentation (`Project_Report.md`). |
| `requirements.txt` | List of all Python dependencies. |

## Setup and Run Instructions

### 1. Prerequisites

*   Python 3.8+
*   `pip`

### 2. Installation

Navigate to the project root directory and install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Data Preparation

The data pipeline script will generate the synthetic dataset, perform feature engineering, and save the processed files.

```bash
python3 src/data_pipeline.py
```

### 4. Model Training

Train the ST-EV-TransGNN model. This will save the trained weights to `trained_models/`.

```bash
python3 src/train.py
```

### 5. Evaluation and Analysis

Generate the final performance metrics and visualizations. The results will be saved in the `results/` directory and copied to `app/static/`.

```bash
python3 src/evaluate.py
```

### 6. Run the Web Application (UI)

Start the FastAPI server:

```bash
python3 app/main.py
```

The application will be accessible at `http://127.0.0.1:8000`. Open this URL in your browser to interact with the forecasting system.
