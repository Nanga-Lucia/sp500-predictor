import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="S&P 500 AI Predictor", layout="wide")

# 1. Check Model
if not os.path.exists('final_sp500_model.pkl'):
    st.error("❌ Model file not found in C:\\SCHOOL PROJECT")
    st.stop()

model = joblib.load('final_sp500_model.pkl')

st.title("📈 S&P 500 Movement Classifier")

# 2. Sidebar
ticker = st.sidebar.text_input("Enter Ticker", "SPY")

st.write("### Status: Fetching Data...") 

try:
    df = yf.download(ticker, period="1y")
    
    # --- THE FIX FOR THE MULTI-COLUMN ERROR ---
    if not df.empty:
        # This flattens the columns so 'Close' is just 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    # ------------------------------------------

    data_ready = False

    if df.empty:
        st.warning("⚠️ Live API connection failed. Using 'Simulation Mode'.")
        simulated_data = {
            'Price_Return': [0.0012], 'Vol_Change': [-0.05], 'Dist_SMA_50': [1.02],
            'Dist_SMA_200': [1.08], 'RSI': [55.4], 'BB_Relative': [0.65]
        }
        latest_row = pd.DataFrame(simulated_data)
        data_ready = True
    else:
        st.write(f"✅ Successfully loaded and flattened {len(df)} rows.")
        
        # Now the calculation will work because 'Close' is a single column
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/loss.replace(0, 0.001))))

        # Bollinger Bands
        std = df['Close'].rolling(window=20).std()
        df['BB_High'] = df['Close'].rolling(window=20).mean() + (std * 2)
        df['BB_Low'] = df['Close'].rolling(window=20).mean() - (std * 2)

        # Stationary Features (Objectives II & III)
        df['Price_Return'] = df['Close'].pct_change()
        df['Vol_Change'] = df['Volume'].pct_change()
        df['Dist_SMA_50'] = df['Close'] / df['SMA_50']
        df['Dist_SMA_200'] = df['Close'] / df['SMA_200']
        df['BB_Relative'] = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low']).replace(0, 0.001)

        # Cleanup & Final Row Selection
        df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
        stationary_features = ['Price_Return', 'Vol_Change', 'Dist_SMA_50', 'Dist_SMA_200', 'RSI', 'BB_Relative']
        
        latest_row = df_clean[stationary_features].tail(1)
        data_ready = True

    # 3. UI Display (Keep this the same as before)
    if data_ready:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Current Market State")
            st.dataframe(latest_row)
            
            if st.button('Run AI Prediction'):
                prediction = model.predict(latest_row)[0]
                result = "UP 🚀" if prediction == 1 else "DOWN 📉"
                st.info(f"### Prediction: {result}")
        
        with col2:
            st.subheader("Price History")
            if not df.empty:
                st.line_chart(df['Close'].tail(50))

except Exception as e:
    st.error(f"❌ An error occurred: {e}")