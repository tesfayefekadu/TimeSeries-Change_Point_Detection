import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import pymc3 as pm

# Load and Clean Data
def load_and_clean_data(file_path):
    df = pd.read_excel(file_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Price']).drop_duplicates(subset=['Date']).sort_values(by='Date').reset_index(drop=True)
    return df

# Exploratory Data Analysis
def plot_time_series(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Price'], label='Brent Oil Price')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('Brent Oil Price Over Time')
    plt.legend()
    plt.show()

def decompose_time_series(df, period=365):
    result = seasonal_decompose(df.set_index('Date')['Price'], model='additive', period=period)
    result.plot()
    plt.show()
    return result

# Annotate Events
def annotate_events(df, events):
    events_df = pd.DataFrame(list(events.items()), columns=['Date', 'Event'])
    events_df['Date'] = pd.to_datetime(events_df['Date'])
    df = pd.merge(df, events_df, on='Date', how='left')
    return df

# ARIMA Model
def arima_forecast(df, order=(5, 1, 0), steps=30):
    model = ARIMA(df['Price'], order=order)
    arima_result = model.fit()
    forecast = arima_result.forecast(steps=steps)
    return arima_result, forecast

# GARCH Model
def garch_forecast(df, p=1, q=1, steps=30):
    garch_model = arch_model(df['Price'], vol='Garch', p=p, q=q)
    garch_result = garch_model.fit(disp="off")
    vol_forecast = garch_result.forecast(horizon=steps).variance[-1:]
    return garch_result, vol_forecast

# Bayesian Changepoint Detection
def bayesian_changepoint_detection(df):
    with pm.Model() as model:
        switchpoint = pm.DiscreteUniform('switchpoint', lower=0, upper=len(df['Price']) - 1)
        early_mean = pm.Normal('early_mean', mu=df['Price'].mean(), sd=10)
        late_mean = pm.Normal('late_mean', mu=df['Price'].mean(), sd=10)
        
        idx = np.arange(len(df['Price']))
        mean = pm.math.switch(switchpoint >= idx, early_mean, late_mean)
        observed = pm.Normal('observed', mu=mean, sd=1, observed=df['Price'])
        
        trace = pm.sample(1000, cores=1)
    
    pm.traceplot(trace)
    plt.show()
    return trace
