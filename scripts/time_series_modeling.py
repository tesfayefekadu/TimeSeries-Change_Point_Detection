
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA

def load_data(price_path, gdp_path, inflation_path, exchange_path):
    data = pd.read_csv(price_path, parse_dates=['Date'], index_col='Date')
    gdp_data = pd.read_csv(gdp_path, parse_dates=['Date'], index_col='Date')
    inflation_data = pd.read_csv(inflation_path, parse_dates=['Date'], index_col='Date')
    exchange_data = pd.read_csv(exchange_path, parse_dates=['Date'], index_col='Date')
    
    data = data.join([gdp_data, inflation_data, exchange_data], how='inner')
    data.dropna(inplace=True)
    return data

def exploratory_data_analysis(data):
    plt.figure(figsize=(14, 8))
    for i, column in enumerate(data.columns[1:], 1): 
        plt.subplot(2, 2, i)
        plt.plot(data['Price'], label='Brent Oil Price', color='blue')
        plt.plot(data[column], label=column, color='orange')
        plt.title(f"Brent Oil Price vs {column}")
        plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("Correlation matrix:")
    print(data.corr())

def fit_arimax_model(data):
    model = ARIMA(data['Price'], exog=data[['GDP', 'Inflation', 'ExchangeRate']], order=(1, 1, 1))
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def fit_var_model(data):
    model = VAR(data)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def forecast_var_model(var_model, data, steps=5):
    forecast = var_model.forecast(data.values[-var_model.k_ar:], steps=steps)
    return forecast


def apply_lstm_model(data):
    """
    Apply LSTM for time series prediction on Brent oil prices.
    """
    # Preprocess data for LSTM
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Price'].values.reshape(-1, 1))

    # Prepare data for LSTM
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32)
    return model
