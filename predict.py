import yfinance as yf
from prophet import Prophet
import streamlit as st
from datetime import datetime, timedelta
import altair as alt


def predict_stock_price(ticker, days):
    # Download stock data
    stock_data = yf.download(ticker, period='max')

    # Prepare data for Prophet
    df = stock_data.reset_index()[['Date', 'Close']].rename({'Date': 'ds', 'Close': 'y'}, axis='columns')

    # Train Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(df)

    # Make future predictions
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # Get last predicted value
    predicted_price = forecast['yhat'].iloc[-1]

    # Return predicted price
    return predicted_price, stock_data, forecast


# Set app title
st.set_page_config(page_title="M-Stocks")

# Display terms of service popup when the user first uses the app
if st.session_state.get('terms_accepted') != True:
    st.error("""
    # Stock Price Predictor - Terms and Conditions

    By using this app, you agree to the following terms and conditions:

    1. Accuracy of Predictions: The predictions provided by the App are based on historical data and statistical models, and may not be accurate or reliable. We make no representations or warranties as to the accuracy, completeness, or reliability of any prediction made by the App. You acknowledge and agree that any reliance on such predictions is at your own risk.

    2. Investment Decisions: The App is provided for informational purposes only, and should not be used as the sole basis for making any investment decisions. You should consult with a qualified financial advisor before making any investment decisions.

    3. Limitation of Liability: To the maximum extent permitted by law, we will not be liable for any damages or losses arising out of or in connection with your use of the App, including any inaccuracies or errors in the predictions made by the App.

    4. Intellectual Property: The App and all content and materials contained therein, including without limitation the software, algorithms, and graphics, are owned by us and are protected by applicable intellectual property laws.

    5. Prohibited Uses: You may not use the App for any illegal or unauthorized purpose, or to violate any laws or regulations. You may not modify, distribute, or create derivative works of the App, or use any data mining, robots, or similar data gathering or extraction methods.

    6. Privacy Policy: We may collect and use certain personal information from you in connection with your use of the App. Our collection and use of such information will be governed by our Privacy Policy, which is incorporated by reference into these terms and conditions.

    7. Termination: We may terminate your use of the App at any time, without notice or liability, for any reason whatsoever.

    8. Governing Law: These terms and conditions will be governed by and construed in accordance with the laws of the jurisdiction in which we are located.

    9. Entire Agreement: These terms and conditions constitute the entire agreement between you and us with respect to the App, and supersede all prior or contemporaneous agreements or understandings, whether written or oral.

    If you have any questions or concerns about these terms and conditions, please contact us at [insert contact information].

    By clicking the button below, you acknowledge that you have read and agree to these terms and conditions.
    """)
    terms_accepted = st.button("Click twice to agree")
    if terms_accepted:
        st.session_state['terms_accepted'] = True
else:
    # Display app contents
    ticker = st.sidebar.text_input("Enter stock ticker symbol (e.g. AAPL)")

    if not ticker:
        st.warning("Please enter a stock ticker symbol.")
    else:
        days = st.sidebar.slider("Number of days into the future to predict", 1, 365)
        
        with st.spinner(f"Predicting {ticker} stock price..."):
            predicted_price, stock_data, forecast = predict_stock_price(ticker, days)
        
        # Display predicted price
        st.success(f"Predicted {ticker} stock price in {days} days: {predicted_price:.2f}$")
        
        # Create line chart of historical and predicted stock prices
        chart_data = stock_data[['Close']].join(forecast.set_index('ds')[['yhat']])
        chart_data = chart_data.rename(columns={'Close': 'Actual', 'yhat': 'Predicted'})
        chart_data = chart_data.reset_index()
        chart_data = chart_data.melt('Date', var_name='Price', value_name='Value')
        chart = alt.Chart(chart_data).mark_line().encode(
            x='Date:T',
            y='Value:Q',
            color='Price:N'
        ).properties(
            width=800,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)
