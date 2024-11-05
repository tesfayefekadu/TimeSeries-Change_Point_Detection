# app.py

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

('', '', '', '')

# Load and process datasets
brent_data = pd.read_csv('../data/BrentOilPrices.csv', parse_dates=['Date'], index_col='Date')
gdp_data = pd.read_csv('../data/syntetic data/GDP.csv', parse_dates=['Date'], index_col='Date')
inflation_data = pd.read_csv('../data/syntetic data/Inflation.csv', parse_dates=['Date'], index_col='Date')
exchange_rate_data = pd.read_csv('../data/syntetic data/ExchangeRate.csv', parse_dates=['Date'], index_col='Date')

# Combine datasets
data = brent_data.join([gdp_data, inflation_data, exchange_rate_data], how='inner').dropna()

# Endpoint to get processed data for frontend display
@app.route('/data', methods=['GET'])
def get_data():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    data_filtered = data[start_date:end_date] if start_date and end_date else data
    return jsonify(data_filtered.reset_index().to_dict(orient="records"))

# Endpoint to get ARIMAX model forecast
@app.route('/forecast/arimax', methods=['GET'])
def forecast_arimax():
    model = ARIMA(data['Price'], exog=data[['GDP', 'Inflation', 'ExchangeRate']], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=10, exog=data[['GDP', 'Inflation', 'ExchangeRate']][-10:])
    return jsonify(forecast.to_list())

# Endpoint to get VAR model forecast
@app.route('/forecast/var', methods=['GET'])
def forecast_var():
    model = VAR(data)
    model_fit = model.fit()
    forecast = model_fit.forecast(data.values[-model_fit.k_ar:], steps=10)
    return jsonify(forecast.tolist())

# Additional route to serve model performance metrics (RMSE, MAE, etc.)
@app.route('/metrics', methods=['GET'])
def get_metrics():
    # This should include calculations of RMSE, MAE, etc.
    metrics = {
        "RMSE": 1.23,  # Example values
        "MAE": 0.98,
    }
    return jsonify(metrics)

if __name__ == "__main__":
    app.run(debug=True)
