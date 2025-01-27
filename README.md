# Stock Market Prediction Using Financial News and LSTM

## Overview

This project predicts stock prices by integrating financial news sentiment analysis and technical indicators. The methodology involves retrieving stock data, processing financial news, performing feature engineering and selection, and using an LSTM model for forecasting.

## Techniques Used and Justifications

### 1. **Data Collection**

- **Yahoo Finance (**``**)**: Fetches historical stock data for Apple (AAPL) to provide a time-series dataset.
- **News API (**``**)**: Retrieves financial news headlines to capture market sentiment.

### 2. **Sentiment Analysis**

- **FinBERT (**``**)**: A specialized NLP model for financial text sentiment analysis. It helps extract market sentiment from news headlines.
- **Softmax Activation (**``**)**: Converts model outputs into probability scores to measure sentiment intensity.

### 3. **Feature Engineering**

- **Technical Indicators (**``**)**: Generates a comprehensive set of indicators to analyze market trends.
- **Lag Features**: Adds past closing prices as new features to capture time dependencies.

### 4. **Feature Selection**

- **Recursive Feature Elimination (RFE) (**``**)**: Selects the most relevant features using a `RandomForestRegressor`, reducing dimensionality and improving model performance.

### 5. **Data Splitting and Scaling**

- **Train-Validation-Test Split (**``**)**: Ensures robust model evaluation.
- **MinMax Scaling (**``**)**: Normalizes the dataset to improve convergence in deep learning models.

### 6. **LSTM Model for Time-Series Forecasting**

- **Sequential Model (**``**)**: Defines the deep learning model architecture.
- **LSTM Layers (**``**)**: Captures temporal dependencies in stock price movements.
- **Dropout Layers (**``**)**: Prevents overfitting by randomly deactivating neurons.
- **Dense Layers (**``**)**: Ensures proper transformation of features.
- **Early Stopping (**``**)**: Stops training when validation loss stops improving.

### 7. **Evaluation and Visualization**

- **Mean Absolute Error (MAE) & Mean Squared Error (MSE) (**``**)**: Measures model accuracy.
- **Matplotlib & Seaborn (**``**, **``**)**: Used for visualizing model performance and stock price trends.

## Execution

Run the script by executing:

```bash
python script.py
```

Ensure dependencies are installed using:

```bash
pip install -r requirements.txt
```

## Results

- The LSTM model forecasts stock prices based on historical and sentiment-enhanced features.
- The results are evaluated with MAE and MSE, with plots showcasing prediction trends.

## Conclusion

By combining financial sentiment analysis with technical indicators, this approach enhances stock market predictions. Further improvements can be made by integrating alternative data sources or experimenting with different deep learning architectures.

