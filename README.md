# Brent Oil Price Analysis Project

## Overview
This project analyzes historical Brent oil prices to understand price trends, volatility, and key factors influencing these prices. Using a combination of time series analysis, econometric models, and machine learning techniques, the project explores correlations with global events, economic indicators, and political factors. It also includes an interactive dashboard built with Flask (backend) and React (frontend) to visualize insights for stakeholders.

## Data Files
Place the following CSV files in the data directory:

BrentOilPrices.csv: Historical Brent oil prices.
GDP.csv, Inflation.csv, ExchangeRate.csv: Economic indicators to analyze their effect on oil prices.

## Features
Data Processing and Exploration: Data cleaning, trend identification, and change point detection.
Modeling: ARIMA and GARCH models for trend and volatility analysis, with options for advanced models such as VAR and LSTM.
Interactive Dashboard: Provides an accessible interface to visualize trends, model results, and the impact of economic indicators on Brent oil prices.


## Key Insights

Historical Trends and Change Points: Significant periods of price volatility align with major global events.
Economic Indicators Correlation: GDP, inflation, and exchange rate fluctuations have notable effects on oil price movements.
Modeling Results: ARIMA and GARCH models reveal key price trends and volatility patterns, and the LSTM model (optional) captures complex dependencies.
## Future Enhancements
Add more indicators, such as unemployment and renewable energy adoption.
Incorporate real-time data feeds for dynamic updates in the dashboard.
Improve model accuracy through advanced time series and machine learning models.
## Dependencies
Python: pandas, matplotlib, ruptures, statsmodels, arch, flask
React: recharts, react-chartjs-2, d3.js, axios