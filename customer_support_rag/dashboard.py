import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load Data ---
df = pd.read_csv("F:/working-projects/customer_support_rag/customer_support_tickets2.csv")

# --- Utility: Clean all numeric-looking columns ---
def clean_numeric_columns(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)')[0]
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

df = clean_numeric_columns(df)

# --- Extra safety: force key metrics to numeric ---
for col in ["First Response Time", "Time to Resolution", "Customer Satisfaction Rating"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# --- Dashboard Title ---
st.title("📊 Customer Support Dashboard")

# --- Key Metrics ---
col1, col2, col3 = st.columns(3)

with col1:
    avg_response = df["First Response Time"].mean() if "First Response Time" in df.columns else 0
    st.metric("⏱ Avg First Response Time", f"{avg_response:.2f}")

with col2:
    avg_resolution = df["Time to Resolution"].mean() if "Time to Resolution" in df.columns else 0
    st.metric("✅ Avg Resolution Time", f"{avg_resolution:.2f}")

with col3:
    avg_satisfaction = df["Customer Satisfaction Rating"].mean() if "Customer Satisfaction Rating" in df.columns else 0
    st.metric("⭐ Avg Satisfaction", f"{avg_satisfaction:.2f}")

# --- FRAME 1: Ticket Distribution Charts ---
st.markdown("### 📦 Ticket Distributions")

colA, colB = st.columns(2)
colC, colD = st.columns(2)

with colA:
    if "Ticket Status" in df.columns:
        st.subheader("📌 Status")
        st.bar_chart(df["Ticket Status"].value_counts())

with colB:
    if "Ticket Priority" in df.columns:
        st.subheader("⚡ Priority")
        st.bar_chart(df["Ticket Priority"].value_counts())

with colC:
    if "Ticket Channel" in df.columns:
        st.subheader("📡 Channel")
        st.bar_chart(df["Ticket Channel"].value_counts())

with colD:
    if "Created Date" in df.columns:
        st.subheader("📅 Over Time")
        df["Created Date"] = pd.to_datetime(df["Created Date"], errors="coerce")
        trend = df.groupby(df["Created Date"].dt.to_period("M")).size()
        st.line_chart(trend)

# --- FRAME 2: Resolution & Correlations ---
st.markdown("### 📈 Resolution & Insights")

colE, colF = st.columns(2)
colG, colH = st.columns(2)

with colE:
    if "Ticket Priority" in df.columns and "Time to Resolution" in df.columns:
        st.subheader("⚡ Resolution by Priority")
        priority_res = df.groupby("Ticket Priority")["Time to Resolution"].mean()
        st.bar_chart(priority_res)

with colF:
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    if not numeric_df.empty:
        st.subheader("📊 Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

with colG:
    if "Customer Satisfaction Rating" in df.columns and "Ticket Channel" in df.columns:
        st.subheader("⭐ Satisfaction by Channel")
        sat_by_channel = df.groupby("Ticket Channel")["Customer Satisfaction Rating"].mean()
        st.bar_chart(sat_by_channel)

with colH:
    if "Customer Satisfaction Rating" in df.columns and "Ticket Priority" in df.columns:
        st.subheader("⭐ Satisfaction by Priority")
        sat_by_priority = df.groupby("Ticket Priority")["Customer Satisfaction Rating"].mean()
        st.bar_chart(sat_by_priority)

# --- Raw Data at Bottom ---
st.markdown("### 📂 Raw Ticket Data")
st.dataframe(df)
