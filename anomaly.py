import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

# Title
st.title("💱 Exchange Rate Anomaly Detector")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_excel("Exchange Rates.xlsx", usecols="D:G")
    df.columns = ["EUR", "GBP", "USD", "MYR"]
    df = df.dropna()
    df = df[df.applymap(lambda x: isinstance(x, (int, float)))]
    return df

df = load_data()

# Train models
@st.cache_resource
def train_models(df):
    models = {}
    for currency in df.columns:
        model = IsolationForest(contamination=0.01, random_state=42)
        model.fit(df[[currency]])
        models[currency] = model
    return models

models = train_models(df)

# Sidebar inputs
st.sidebar.header("📥 Enter Today's Exchange Rates")
user_input = {}
for currency in df.columns:
    user_input[currency] = st.sidebar.number_input(f"{currency}", min_value=0.0, format="%.4f")

# Predict anomalies
user_df = pd.DataFrame([user_input])
anomalies = {cur: models[cur].predict(user_df[[cur]])[0] for cur in df.columns}
anomalous = [cur for cur, pred in anomalies.items() if pred == -1]

# Display results
st.subheader("🔍 Anomaly Detection Result")
if anomalous:
    st.error("⚠️ Anomalies detected in:")
    for cur in anomalous:
        st.write(f"- {cur}: {user_input[cur]}")
else:
    st.success("✅ No anomalies detected in the entered exchange rates.")

# Optional: Show historical data
with st.expander("📊 Show Historical Exchange Rates"):
    st.dataframe(df.describe().T)
