# 📈 AI-Powered Stock Market Prediction Web Application

An end-to-end **Deep Learning-based stock prediction system** using **LSTM, GRU, CNN-LSTM + SHAP Explainability**.

---

## 🚀 Project Overview

This project predicts stock prices using deep learning models and provides:

- 📊 Accurate stock price prediction  
- 🔄 Multi-stock learning  
- 📈 Interactive visualizations  
- 🔍 Explainable AI (SHAP)  

---

## 🧠 Models Used

- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- CNN-LSTM (Hybrid Model)

---

## ⚙️ Workflow

### 🔹 System Pipeline

1. **Data Collection**
   - Fetch historical stock data using `yfinance`

2. **Data Preprocessing**
   - Handle missing values  
   - Normalize data  
   - Feature selection (closing price)

3. **Sequence Creation**
   - Convert time-series into supervised format  
   - Apply sliding window technique  

4. **Model Training**
   - Train LSTM, GRU, CNN-LSTM models  
   - Use training & validation datasets  

5. **Model Evaluation**
   - Metrics: MSE, RMSE, MAE, R² Score  

6. **Prediction**
   - Generate future stock prices  
   - Predict direction (Up/Down)

7. **Explainability**
   - Apply SHAP to understand feature importance  

---

---

## 📊 Results & Outputs

> ⚠️ NOTE: Images are loaded from GitHub using raw links

---

### 📉 Actual vs Predicted Prices

![Actual vs Predicted](https://raw.githubusercontent.com/Bijoy781999/AI-Powered-Stock-Prediction-Web-Application/main/OutPuts/Actual%20vs%20Predicted.png)

- Strong alignment between predicted & actual prices  
- Captures trends effectively  

---

### 📊 Multi-Stock Closing Price Trend

![Closing Trend](https://raw.githubusercontent.com/Bijoy781999/AI-Powered-Stock-Prediction-Web-Application/main/OutPuts/Closing%20Price%20Trend.png)

- Shows multiple stock behavior  
- Improves generalization  

---

### 🔄 Direction Prediction

![Direction](https://raw.githubusercontent.com/Bijoy781999/AI-Powered-Stock-Prediction-Web-Application/main/OutPuts/Direction%20Movement.png)

- Predicts up/down movement  
- ~50–55% accuracy  

---

### 📊 Model Comparison

![Model Comparison](https://raw.githubusercontent.com/Bijoy781999/AI-Powered-Stock-Prediction-Web-Application/main/OutPuts/Model%20Comparison.png)

- GRU best for single stock  
- LSTM best for multi-stock  

---

### 📉 Prediction Error Density

![Error Density](https://raw.githubusercontent.com/Bijoy781999/AI-Powered-Stock-Prediction-Web-Application/main/OutPuts/Prediction%20Error%20Density.png)

- Errors centered around zero  
- Stable predictions  

---

### 📉 Residual Errors

![Residual](https://raw.githubusercontent.com/Bijoy781999/AI-Powered-Stock-Prediction-Web-Application/main/OutPuts/Residual%20Error.png)

- Shows volatility spikes  
- No major bias  

---

### 📈 Rolling Directional Accuracy

![Rolling Accuracy](https://raw.githubusercontent.com/Bijoy781999/AI-Powered-Stock-Prediction-Web-Application/main/OutPuts/Rolling%20Direction.png)

- Fluctuates due to market volatility  

---

### 🔍 SHAP Explainability

![SHAP](https://raw.githubusercontent.com/Bijoy781999/AI-Powered-Stock-Prediction-Web-Application/main/OutPuts/SHAP%20Summary.png)

- Shows feature importance  
- Improves interpretability  

---

### 📉 Training vs Validation Loss

![Loss](https://raw.githubusercontent.com/Bijoy781999/AI-Powered-Stock-Prediction-Web-Application/main/OutPuts/Training%20and%20Validation.png)

- Smooth convergence  
- Minimal overfitting  

---

## 📌 Key Insights

- ✅ GRU performs best for single-stock prediction  
- ✅ LSTM performs best for multi-stock prediction  
- 🔍 SHAP improves model interpretability  
- ⚠️ Direction prediction remains challenging (~50%)  

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- yfinance  
- SHAP  
- Flask  

---

## 📂 Project Structure
