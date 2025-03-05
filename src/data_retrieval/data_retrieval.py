import yfinance as yf
import pandas as pd

def get_spy_daily_closing_prices(start_date, end_date):
    spy_data = yf.download('SPY', start=start_date, end=end_date, interval='1d')
    
    closing_prices = spy_data[[('Close', 'SPY')]]
    closing_prices.columns = ['Close']
    return closing_prices

if __name__ == "__main__":
    start_date = '2021-01-01'
    end_date = '2024-01-01'
    
    closing_prices = get_spy_daily_closing_prices(start_date, end_date)
    
    closing_prices.to_csv('data/raw/spy_daily_closing_prices.csv')