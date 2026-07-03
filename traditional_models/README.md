# 📈 Traditional Deep Learning Models for Stock Market Prediction

> **Comparative Analysis of LSTM, GRU, and CNN-LSTM Architectures for Multi-Stock Price Forecasting**

---

## 📌 Project Overview

This module presents a comparative study of three widely used deep learning architectures for stock market forecasting.

The objective is to predict future stock prices by learning historical market trends from multiple companies while evaluating the strengths and weaknesses of different sequential deep learning models.

The project includes complete data preprocessing, feature engineering, model training, evaluation, explainability, and performance comparison.

---

# ✨ Features

- 📈 Historical stock price prediction
- 🧠 Multiple Deep Learning architectures
- 📊 Multi-stock training
- 📉 Technical indicator feature engineering
- 📈 Comprehensive model comparison
- 📊 Regression performance evaluation
- 🔄 Directional movement analysis
- 📉 Residual error analysis
- 📊 Prediction error density analysis
- 📈 Rolling directional accuracy
- 🔍 SHAP explainability
- 📚 Complete training notebook

---

# 🧠 Deep Learning Models

## LSTM (Long Short-Term Memory)

LSTM is designed to capture long-term dependencies in sequential financial data by utilizing gated memory cells.

### Characteristics

- Long-term dependency learning
- Memory cell architecture
- Suitable for sequential forecasting
- Stable training on time-series data

---

## GRU (Gated Recurrent Unit)

GRU simplifies the LSTM architecture while maintaining strong forecasting performance through update and reset gates.

### Characteristics

- Faster training
- Lower computational complexity
- Reduced number of parameters
- Efficient sequence learning

---

## CNN-LSTM

CNN-LSTM combines convolutional feature extraction with temporal sequence modeling.

### Characteristics

- Local pattern extraction
- Temporal dependency learning
- Hybrid deep learning architecture
- Improved feature representation

---

# 📂 Project Structure

```text
traditional_models/
│
├── Evolution_stock_prediction.ipynb
├── tickers.csv
│
├── sample_dataset/
│
├── saved_models/
│
├── results/
│   ├── Actual_vs_Predicted.png
│   ├── Closing_Price_Trend.png
│   ├── Direction_Movement.png
│   ├── Model_Comparison.png
│   ├── Prediction_Error_Density.png
│   ├── Residual_Error.png
│   ├── Rolling_Direction.png
│   ├── SHAP_Summary.png
│   └── Training_and_Validation.png
│
└── README.md
```

---

# 📊 Dataset

The project uses historical stock market data collected from **Yahoo Finance**.

The dataset contains:

- Open Price
- High Price
- Low Price
- Close Price
- Adjusted Close
- Trading Volume

Multiple stocks are combined into a single dataset to improve model generalization.

A sample dataset is included in this repository for demonstration purposes.

---

# ⚙️ Data Preprocessing

The following preprocessing techniques are applied before model training:

- Missing value handling
- Data normalization using Min-Max Scaling
- Sliding window sequence generation
- Multi-stock data integration
- Training and testing split

---

# 📈 Model Training Pipeline

The workflow consists of the following stages:

1. Data Collection
2. Data Cleaning
3. Feature Scaling
4. Sequence Generation
5. Model Training
6. Model Validation
7. Performance Evaluation
8. Model Comparison
9. Explainability Analysis

---

# 📊 Evaluation Metrics

The models are evaluated using standard regression metrics.

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Directional Accuracy

These metrics provide both prediction accuracy and trend prediction performance.

---

# 📉 Performance Analysis

The `results/` directory contains various evaluation outputs generated during experimentation.

### Prediction Analysis

- Actual vs Predicted Price
- Closing Price Trend

### Error Analysis

- Residual Error
- Prediction Error Density

### Trend Analysis

- Directional Movement
- Rolling Directional Accuracy

### Model Comparison

- LSTM vs GRU vs CNN-LSTM
- MSE
- RMSE
- MAE
- R² Score

### Explainability

- SHAP Feature Importance
- SHAP Interaction Summary

### Training Analysis

- Training Loss
- Validation Loss

---

# 📁 Results Folder

The `results/` folder contains the generated evaluation figures, including:

- Prediction visualizations
- Error analysis
- Performance comparison
- Training curves
- SHAP explainability plots

These results provide a comprehensive evaluation of all implemented models.

---

# 🚀 How to Run

## Clone the repository

```bash
git clone <repository-url>
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Launch Jupyter Notebook

```bash
jupyter notebook
```

## Open

```text
Evolution_stock_prediction.ipynb
```

Run all notebook cells to reproduce the complete workflow.

---

# 🛠 Technologies Used

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Keras

### Data Processing

- NumPy
- Pandas
- Scikit-learn

### Data Collection

- Yahoo Finance (yfinance)

### Visualization

- Matplotlib
- SHAP

---

# 🔮 Future Improvements

- Transformer-based forecasting models
- Attention-enhanced LSTM
- Hybrid CNN-Transformer architecture
- Explainable AI with LIME
- Hyperparameter optimization
- Multi-step forecasting
- Real-time prediction
- Web application deployment

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
