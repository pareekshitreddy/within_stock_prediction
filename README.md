# Stock Market Prediction Using Financial News and LSTM

## Overview

This project predicts stock prices by integrating financial news sentiment analysis and technical indicators. The methodology involves retrieving stock data, processing financial news, performing feature engineering and selection, and using an LSTM model for forecasting.

## Techniques Used and Justifications

### 1. **Data Collection**

- **Yahoo Finance (`yfinance`)**: Fetches historical stock data for Apple (AAPL) to provide a time-series dataset.
- **News API (`requests`)**: Retrieves financial news headlines to capture market sentiment.

### 2. **Sentiment Analysis**

- **FinBERT (`transformers`)**: A specialized NLP model for financial text sentiment analysis. It helps extract market sentiment from news headlines.
- **Softmax Activation (`torch.nn.functional.softmax`)**: Converts model outputs into probability scores to measure sentiment intensity.

### 3. **Feature Engineering**

- **Technical Indicators (`ta`)**: Generates a comprehensive set of indicators to analyze market trends.
- **Lag Features**: Adds past closing prices as new features to capture time dependencies.

### 4. **Feature Selection**

- **Recursive Feature Elimination (RFE) (`sklearn.feature_selection.RFE`)**: Selects the most relevant features using a `RandomForestRegressor`, reducing dimensionality and improving model performance.

### 5. **Data Splitting and Scaling**

- **Train-Validation-Test Split (`sklearn.model_selection.train_test_split`)**: Ensures robust model evaluation.
- **MinMax Scaling (`sklearn.preprocessing.MinMaxScaler`)**: Normalizes the dataset to improve convergence in deep learning models.

### 6. **LSTM Model for Time-Series Forecasting**

- **Sequential Model (`tensorflow.keras.models.Sequential`)**: Defines the deep learning model architecture.
- **LSTM Layers (`tensorflow.keras.layers.LSTM`)**: Captures temporal dependencies in stock price movements.
- **Dropout Layers (`tensorflow.keras.layers.Dropout`)**: Prevents overfitting by randomly deactivating neurons.
- **Dense Layers (`tensorflow.keras.layers.Dense`)**: Ensures proper transformation of features.
- **Early Stopping (`tensorflow.keras.callbacks.EarlyStopping`)**: Stops training when validation loss stops improving.

### 7. **Evaluation and Visualization**

- **Mean Absolute Error (MAE) & Mean Squared Error (MSE) (`sklearn.metrics`)**: Measures model accuracy.
- **Matplotlib & Seaborn (`matplotlib.pyplot`, `seaborn`)**: Used for visualizing model performance and stock price trends.

## How to Get and Use Your News API Key

To retrieve financial news, you need an API key from NewsAPI. Follow these steps:

1. Visit [NewsAPI](https://newsapi.org/).
2. Sign up for a free account.
3. Navigate to your account settings and generate an API key.
4. Replace `your_api_key_here` in the script with your generated API key:
   ```python
   url = "https://newsapi.org/v2/everything?q=stock%20market&apiKey=your_api_key_here"
   response = requests.get(url)
   news_data = response.json()
   ```

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

