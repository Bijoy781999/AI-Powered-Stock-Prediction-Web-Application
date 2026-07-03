# 🚀 Advanced AI Stock Market Prediction

> **Advanced Deep Learning-Based Stock Market Forecasting using TimeMachine (State-Space 4-Mamba) and iTransformer with an Interactive FastAPI Web Application**

---

## 📌 Project Overview

This module presents an advanced stock market forecasting system that leverages modern deep learning architectures for next-day stock price prediction.

Unlike traditional recurrent neural networks, this implementation utilizes:

* **TimeMachine (State-Space 4-Mamba Model)**
* **iTransformer (Inverted Transformer)**

The trained model is deployed using **FastAPI**, while the frontend is developed with **HTML**, **CSS**, and **JavaScript**, providing an interactive dashboard for real-time prediction and visualization.

---

# ✨ Features

* 📈 Predicts future stock prices using advanced AI models
* ⚡ Automatically loads the best-performing trained model
* 🌍 Supports live market data from Yahoo Finance
* 💾 Falls back to local datasets when live data is unavailable
* 📊 Interactive visualization dashboard
* 📉 Residual error analysis
* 📈 Rolling Direction Accuracy visualization
* 📊 Prediction Error Density Histogram
* 🔄 Actual vs Predicted Direction comparison
* 🎯 Automatic model selection based on saved metadata
* 🌐 FastAPI REST API backend
* 🎨 Responsive modern user interface

---

# 🧠 Deep Learning Models

## TimeMachine (State-Space 4-Mamba)

A modern State-Space Model (SSM) architecture designed for long-sequence forecasting.

Key Components:

* Reversible Instance Normalization (RevIN)
* Mamba Blocks
* RMS Normalization
* Selective Scan Mechanism
* Residual Learning

---

## iTransformer

An Inverted Transformer architecture specialized for time-series forecasting.

Key Components:

* RevIN Normalization
* Transformer Encoder
* Multi-Head Self Attention
* Sequence Projection
* Residual Learning

---

# 📂 Project Structure

```text
advanced_models/
│
├── backend.py
├── stock-market-forecasting.ipynb
├── index.html
├── style.css
├── app.js
│
├── data/
│   ├── 7000Stocks_all.csv
│   ├── NIFTY50_all.csv
│   └── US500_all.csv
│
├── saved_models/
│   ├── TimeMachine_delta_best.pth
│   ├── iTransformer_delta_best.pth
│   └── best_model_info.json
│
├── results/
│   ├── Prediction Graphs
│   ├── Error Analysis
│   ├── Accuracy Plots
│   └── Evaluation Metrics
│
├── screenshots/
│
└── README.md
```

---

# 📊 Datasets

The project uses multiple stock market datasets.

* US500 Stocks
* NIFTY50 Stocks
* 7000Stocks Dataset

For live prediction, stock prices are fetched directly from **Yahoo Finance**.

When live data is unavailable, the application automatically switches to the local historical dataset.

---

# ⚙️ Feature Engineering

The following technical features are generated before prediction:

* Close Price Scaling
* Log Returns
* Momentum (3, 5, 10)
* Moving Average Difference
* Rolling Mean
* Rolling Standard Deviation
* RSI (14)

---

# 📈 Performance Evaluation

The application provides multiple evaluation plots:

* ✅ Actual vs Predicted Price
* ✅ Residual Error Analysis
* ✅ Rolling Direction Accuracy
* ✅ Prediction Error Density
* ✅ Actual vs Predicted Direction

These plots help evaluate both regression performance and directional forecasting capability.

---

# 🌐 Web Application

The project includes a complete web application.

### Backend

* FastAPI
* REST API
* Dynamic Model Loading
* Live Data Retrieval

### Frontend

* HTML5
* CSS3
* JavaScript
* Chart.js

Users can:

* Search stocks
* Select datasets
* View predictions
* Analyze model performance
* Explore interactive charts

---

# 🚀 How to Run

## 1. Clone the repository

```bash
git clone <https://github.com/Bijoy781999/AI-Powered-Stock-Prediction-Web-Application>
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Start the FastAPI server

```bash
python backend.py
```

or

```bash
uvicorn backend:app --reload
```

## 4. Open your browser

```
http://127.0.0.1:8000
```

---

# 📷 Results

The `results/` folder contains:

* Prediction graphs
* Training results
* Error analysis
* Direction accuracy plots
* Model comparison figures

The `screenshots/` folder contains images of the deployed web application.

---

# 🛠 Technologies Used

### Programming Language

* Python
* JavaScript
* HTML5
* CSS3

### Deep Learning

* PyTorch

### Backend

* FastAPI
* Uvicorn

### Data Processing

* NumPy
* Pandas
* Scikit-learn

### Data Source

* Yahoo Finance (yfinance)

### Visualization

* Matplotlib
* Chart.js

---

# 🔮 Future Improvements

* Multi-step forecasting
* Portfolio recommendation
* Sentiment analysis integration
* Explainable AI (SHAP/LIME)
* Docker deployment
* Cloud deployment
* Mobile application
* User authentication
* Real-time WebSocket updates

---

# 👨‍💻 Authors

**Bijoy Bhadra**

B.Tech in Computer Science & Technology

JIS College of Engineering

---

# 📄 License

This project is developed for academic and research purposes.

---

⭐ If you found this project useful, consider giving the repository a **Star**.

