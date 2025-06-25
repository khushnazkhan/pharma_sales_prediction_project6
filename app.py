import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

#  Page settings
st.set_page_config(page_title="Rossmann Store Sales Prediction", layout="wide")
st.title("Rossmann Store Sales Prediction Dashboard")

# Set base directory (where this file is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set model and feature paths (same folder)
MODEL_PATH = os.path.join(BASE_DIR, "ml_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features_list.pkl")

# Check if files exist
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file (ml_model.pkl) not found in this folder.")
    st.stop()

if not os.path.exists(FEATURES_PATH):
    st.error("❌ Feature list file (features_list.pkl) not found in this folder.")
    st.stop()

#  Load model and features
model = joblib.load(MODEL_PATH)
feature_list = joblib.load(FEATURES_PATH)

st.success("Model and features loaded successfully!")

# UI Inputs
store_id = st.text_input("Enter Store ID")
uploaded_file = st.file_uploader(" Upload CSV (must include Date,Store,IsWeekend, IsHoliday, IsPromo)", type="csv")

if uploaded_file and store_id:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])

    # Feature Engineering (same as training)
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['IsWeekend'] = (df['Date'].dt.weekday >= 5).astype(int)
    df['IsHoliday'] = df['StateHoliday'].apply(lambda x: 0 if str(x) == '0' else 1)
    df['PromoInterval'] = df.get('PromoInterval', 'None').astype('category').cat.codes
    df['StoreType'] = df.get('StoreType', 'a').astype('category').cat.codes
    df['Assortment'] = df.get('Assortment', 'a').astype('category').cat.codes
    df['Open'] = df.get('Open', 1).fillna(1)
    df['IsPromo'] = df.get('Promo', 0)

    # Fill missing required features
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0

    # Match column order and convert to NumPy
    X = df[feature_list].values

    # Predict
    preds = model.predict(X)
    df['Predicted_Sales'] = preds.clip(min=0)

    # Show Table
    st.subheader("📊 Prediction Results")
    st.dataframe(df[['Date', 'Predicted_Sales']])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Date'], df['Predicted_Sales'], marker='o', color='blue')
    ax.set_title(f" Predicted Sales for Store {store_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Sales")
    ax.grid(True)
    st.pyplot(fig)

    # Download CSV
    output = pd.DataFrame({
        'Id': range(1, len(preds) + 1),
        'Sales': preds.clip(min=0)
    })
    csv = output.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Prediction CSV", csv, "predicted_sales.csv", "text/csv")
