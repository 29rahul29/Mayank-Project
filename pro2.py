from nsepy import get_history
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def fetch_nse_data(stock, start_date, end_date):
    data = get_history(symbol=stock, start=start_date, end=end_date)
    data['Return'] = data['Close'].pct_change()
    data['Moving Average'] = data['Close'].rolling(window=20).mean()
    data['Volatility'] = data['Return'].rolling(window=20).std()
    return data

def plot_stock_trend(data, stock):
    plt.figure(figsize=(14, 7))
    plt.title(f"{stock} Price Trends", fontsize=16)
    plt.plot(data['Close'], label="Close Price", color="blue")
    plt.plot(data['Moving Average'], label="Moving Average (20 Days)", color="orange")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def feature_engineering(data):
    data = data.dropna()
    features = data[['Moving Average', 'Volatility', 'Volume']].copy()
    target = data['Close'].shift(-1)
    return features[:-1], target[:-1]

def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(f"Root Mean Squared Error: {rmse:.2f}")
    return model, X_test, y_test, predictions

def plot_predictions(y_test, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual Prices", color="blue")
    plt.plot(predictions, label="Predicted Prices", color="orange")
    plt.title("Model Predictions vs Actual Prices", fontsize=14)
    plt.xlabel("Test Sample Index")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    stock = "INFY"
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 1, 1)
    
    data = fetch_nse_data(stock, start_date, end_date)
    print(data.head())
    
    plot_stock_trend(data, stock)
    
    features, target = feature_engineering(data)
    
    model, X_test, y_test, predictions = train_model(features, target)
    
    plot_predictions(y_test, predictions)
