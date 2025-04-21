import yfinance as yf
import pandas as pd
from arch import arch_model
from config import START_DATE_TRAIN, START_DATE_VAL, START_DATE_TEST, END_DATE_TRAIN, END_DATE_VAL, END_DATE_TEST, OUT_DIR_TRAIN, OUT_DIR_VAL, OUT_DIR_TEST

def get_spy_daily_closing_prices(start_date, end_date):
    spy_data = yf.download('SPY', start=start_date, end=end_date, interval='1d', auto_adjust=True)
    
    closing_prices = spy_data[[('Close', 'SPY')]]
    closing_prices.columns = ['Close']
    return closing_prices

def calculate_volatility_gjr_garch(prices):
    returns = prices['Close'].pct_change().dropna() * 100 
    gjr_garch_model = arch_model(returns, vol='Garch', p=1, q=1, o=1, dist='t')
    gjr_garch_fit = gjr_garch_model.fit(disp="off")
    volatility = gjr_garch_fit.conditional_volatility
    return volatility


def main(start_date, end_date, out_dir):
    closing_prices = get_spy_daily_closing_prices(start_date, end_date)
    closing_prices.index = pd.to_datetime(closing_prices.index)
    vol = calculate_volatility_gjr_garch(closing_prices)
    closing_prices["vol"] = vol
    closing_prices = closing_prices.dropna()

    closing_prices.to_csv(out_dir)


if __name__ == "__main__":
    main(START_DATE_TRAIN, END_DATE_TRAIN, OUT_DIR_TRAIN)
    main(START_DATE_VAL, END_DATE_VAL, OUT_DIR_VAL)
    main(START_DATE_TEST, END_DATE_TEST, OUT_DIR_TEST)
    
    