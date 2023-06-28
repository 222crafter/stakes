import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

def plot_stock_prices(symbol, days, prediction_days):
    stock = yf.Ticker(symbol)
    history = stock.history(period='1d', interval='1d', start=None, end=None)
    prices = history['Close'].values

    actual_days = days + prediction_days
    actual_prices = prices[-actual_days:-prediction_days]
    predicted_prices = prices[-prediction_days:]

    plt.figure(figsize=(12, 6))
    plt.plot(range(actual_days), actual_prices, label='Actual')
    plt.plot(range(days, actual_days), predicted_prices, label='Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title(f'{symbol} Price Trend')
    plt.legend()
    plt.grid(True)

    return plt

def main():
    st.title("Stock and Cryptocurrency Price Predictor")
    symbol = st.text_input("Enter stock or cryptocurrency symbol", value='AAPL')
    days = st.slider("Number of actual trend days to train AI with (The higher the better)", min_value=1, max_value=365, value=60)
    prediction_days = st.slider("Number of days for prediction trend", min_value=1, max_value=30, value=7)

    if st.button("Generate Chart"):
        chart = plot_stock_prices(symbol, days, prediction_days)
        st.pyplot(chart)

if __name__ == '__main__':
    main()
