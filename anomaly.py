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

# Arbitrage Recommender
import numpy as np

# Build exchange rate matrix from user input
rates = user_input
currencies = list(rates.keys())

# Create pairwise exchange rates
exchange_matrix = pd.DataFrame(index=currencies, columns=currencies)
for from_cur in currencies:
    for to_cur in currencies:
        if from_cur == to_cur:
            exchange_matrix.loc[from_cur, to_cur] = 1.0
        else:
            exchange_matrix.loc[from_cur, to_cur] = rates[to_cur] / rates[from_cur]

# Simulate triangular/quadrilateral loops
def find_arbitrage_paths(matrix):
    paths = []
    for a in currencies:
        for b in currencies:
            for c in currencies:
                for d in currencies:
                    if len({a, b, c, d}) == 4:
                        product = (
                            matrix.loc[a, b] *
                            matrix.loc[b, c] *
                            matrix.loc[c, d] *
                            matrix.loc[d, a]
                        )
                        if product > 1.01:  # Threshold for arbitrage
                            paths.append((a, b, c, d, a, round(product, 4)))
    return paths

arbitrage_paths = find_arbitrage_paths(exchange_matrix)

# Display arbitrage suggestions
st.subheader("💡 Arbitrage Opportunities")
if anomalous and arbitrage_paths:
    st.warning("Anomalies detected. Potential arbitrage paths:")
    for path in arbitrage_paths:
        st.write(f"→ {' → '.join(path[:-1])} | Profit multiplier: {path[-1]}")
elif anomalous:
    st.info("Anomalies detected, but no arbitrage paths found.")
else:
    st.success("No anomalies or arbitrage opportunities detected.")
