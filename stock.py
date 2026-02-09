import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st

# Function to fetch stock data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y", interval="1d")
    return data

# Function to prepare data for LSTM model
def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']].values)
    
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return train_test_split(X, y, test_size=0.2, shuffle=False), scaler

# Function to build and train LSTM model
def build_and_train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=1)
    
    return model

# Function to predict stock prices
def predict_stock_price(model, X_test, scaler):
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    return predicted_stock_price


st.title("📈 Real-Time Stock Market Analysis and Prediction")
st.markdown("Enter a stock ticker symbol to analyze and predict its price using an LSTM model.")

# Input ticker symbol
ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL):", "AAPL")

if st.button("Fetch and Analyze"):
    stock_data = fetch_stock_data(ticker)
    st.subheader(f"📊 {ticker} Stock Data")
    st.write(stock_data.tail())
    
    (X_train, X_test, y_train, y_test), scaler = prepare_data(stock_data)
    
    model = build_and_train_model(X_train, y_train)
    predicted_stock_price = predict_stock_price(model, X_test, scaler)
    
    # Plot actual vs predicted prices
    st.subheader("🔍 Predicted vs Actual Stock Prices")
    fig, ax = plt.subplots()
    ax.plot(stock_data.index[-len(predicted_stock_price):], predicted_stock_price, color='red', label='Predicted Stock Price')
    ax.plot(stock_data.index[-len(predicted_stock_price):], stock_data['Close'][-len(predicted_stock_price):], color='blue', label='Actual Stock Price')
    ax.set_title(f'{ticker} Stock Price Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

    # Additional plots
    st.subheader("📅 Stock Price Trends")
    fig, ax = plt.subplots()
    sns.lineplot(data=stock_data, x=stock_data.index, y="Close", ax=ax, label='Close Price')
    ax.set_title(f'{ticker} Stock Price Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("📈 Stock Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(stock_data['Close'], kde=True, ax=ax, color='purple')
    ax.set_title(f'{ticker} Stock Price Distribution')
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # 3D Scatter Plot
    st.subheader("📊 3D Scatter Plot of Stock Data")
    fig = px.scatter_3d(stock_data.reset_index(), x='Date', y='Close', z='Volume', color='Close', title="3D Scatter Plot of Stock Data")
    st.plotly_chart(fig)

    st.subheader("🟦 Candlestick Chart")
    fig, ax = plt.subplots()
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=stock_data.index,
                                 open=stock_data['Open'],
                                 high=stock_data['High'],
                                 low=stock_data['Low'],
                                 close=stock_data['Close'], name='Candlestick'))

    fig.update_layout(title=f'{ticker} Candlestick Chart',
                      yaxis_title='Price',
                      xaxis_title='Date')

    st.plotly_chart(fig, use_container_width=True)
