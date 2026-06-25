import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Page Configuration
st.set_page_config(
    page_title="Retail Analytics Platform",
    page_icon="📊",
    layout="wide",
)

# Load Data & Model
@st.cache_data
def load_data():
    df = pd.read_csv("clean_transactions.csv")
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    features_df = pd.read_csv("clv_feature_matrix.csv")
    return df, features_df

try:
    df, features_df = load_data()
    model = joblib.load('best_clv_model.pkl')
    model_cols = joblib.load('model_columns.pkl')
except Exception as e:
    st.error(f"Error loading files. Ensure clean_transactions.csv, clv_feature_matrix.csv, and model files exist. Details: {e}")
    st.stop()

# Title
st.title("📊 Retail Analytics Platform")
st.subheader("Data Insights, Customer Segmentation & CLV Prediction Dashboard")

# Sidebar navigation
page = st.sidebar.selectbox("Choose Dashboard Page", ["Executive Overview", "CLV Predictor Tool", "Customer Cohorts"])

# PAGE 1: EXECUTIVE OVERVIEW
if page == "Executive Overview":
    st.header("Executive Overview")
    
    # High Level KPI Cards
    total_sales = df['TransactionAmount'].sum()
    total_tx = df['TransactionID'].nunique()
    total_cust = df['CustomerID'].nunique()
    avg_order = total_sales / total_tx
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${total_sales:,.2f}")
    col2.metric("Total Transactions", f"{total_tx:,}")
    col3.metric("Active Customers", f"{total_cust:,}")
    col4.metric("Avg Order Value", f"${avg_order:.2f}")
    
    st.markdown("---")
    
    # Plots Layout
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Sales by Product Category")
        cat_sales = df.groupby('ProductCategory')['TransactionAmount'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=cat_sales, x='ProductCategory', y='TransactionAmount', palette='viridis', ax=ax)
        plt.xticks(rotation=45)
        plt.ylabel("Total Sales ($)")
        st.pyplot(fig)
        
    with c2:
        st.subheader("Monthly Revenue Trends (Seasonality)")
        df['YearMonth'] = df['TransactionDate'].dt.to_period('M')
        monthly_sales = df.groupby('YearMonth')['TransactionAmount'].sum().reset_index()
        monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=monthly_sales, x='YearMonth', y='TransactionAmount', marker='o', color='purple', ax=ax)
        plt.xticks(rotation=45)
        plt.ylabel("Sales ($)")
        st.pyplot(fig)

# PAGE 2: CLV PREDICTOR TOOL
elif page == "CLV Predictor Tool":
    st.header("🔮 Interactive Customer Lifetime Value Predictor")
    st.write("Input a customer's purchasing history metrics below to predict their purchase amount for the next 180 days.")
    
    # Input sliders
    col1, col2 = st.columns(2)
    with col1:
        recency = st.slider("Recency (Days since last purchase)", 0, 365, 30)
        frequency = st.number_input("Frequency (Number of past transactions in last year)", min_value=1, max_value=100, value=5)
        avg_spend = st.number_input("Average Order Value ($)", min_value=5.0, max_value=1000.0, value=75.0)
    
    with col2:
        tier = st.selectbox("Customer Tier", ["Bronze", "Silver", "Gold", "Platinum"])
        total_spend = frequency * avg_spend
        st.info(f"Calculated Total Spend: ${total_spend:,.2f}")
        
    # Process user inputs to match model format
    input_data = {
        'RecencyDays': recency,
        'Frequency': frequency,
        'TotalSpend': total_spend,
        'AvgSpend': avg_spend,
        'CustomerTier_Gold': 1 if tier == "Gold" else 0,
        'CustomerTier_Platinum': 1 if tier == "Platinum" else 0,
        'CustomerTier_Silver': 1 if tier == "Silver" else 0,
    }
    
    # Create DF matching training column order
    input_df = pd.DataFrame([input_data])
    input_df = input_df[model_cols]
    
    if st.button("Predict 180-Day Value"):
        prediction = model.predict(input_df)[0]
        st.success(f"### Predicted Spending in Next 6 Months: ${prediction:,.2f}")
        st.balloons()

# PAGE 3: CUSTOMER COHORTS
elif page == "Customer Cohorts":
    st.header("👥 Customer Segmentation Analysis")
    st.write("Visualizing demographics and customer tiers distribution.")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=features_df, x='CustomerTier', palette='Set2', ax=ax)
    plt.title("Distribution of Customers Across Tiers")
    plt.xlabel("Loyalty Tier")
    plt.ylabel("Number of Customers")
    st.pyplot(fig)