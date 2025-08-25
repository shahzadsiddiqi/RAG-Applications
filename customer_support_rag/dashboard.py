import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===============================
# 📌 1. Load & Clean Dataset
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("F:/working-projects/RAG-Applications/customer_support_rag/customer_support_tickets2.csv")

    # Clean numeric columns to avoid errors
    numeric_cols = ["Resolution Time (hours)", "Customer Satisfaction (1–5)", "Response Time (minutes)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

df = load_data()

# Page title
st.title("📊 Customer Support Ticket Dashboard")
st.markdown("Analyze ticket trends, satisfaction, and resolution metrics.")

# ===============================
# 📌 2. Filter-based Query Interface
# ===============================
st.markdown("## 🔍 Filter Tickets")

# Dropdowns for filtering
status_filter = st.selectbox("Filter by Status", ["All"] + sorted(df["Ticket Status"].dropna().unique()))
priority_filter = st.selectbox("Filter by Priority", ["All"] + sorted(df["Ticket Priority"].dropna().unique()))
channel_filter = st.selectbox("Filter by Channel", ["All"] + sorted(df["Ticket Channel"].dropna().unique()))

# Apply filters
filtered_df = df.copy()
if status_filter != "All":
    filtered_df = filtered_df[filtered_df["Ticket Status"] == status_filter]
if priority_filter != "All":
    filtered_df = filtered_df[filtered_df["Ticket Priority"] == priority_filter]
if channel_filter != "All":
    filtered_df = filtered_df[filtered_df["Ticket Channel"] == channel_filter]

# Display filtered results
st.markdown("### 🎯 Filtered Tickets")
st.dataframe(filtered_df)

# ===============================
# 📌 3. Visual Analytics
# ===============================
st.markdown("## 📈 Visual Analytics")

col1, col2 = st.columns(2)

# Chart 1: Status Distribution
with col1:
    st.markdown("### Ticket Status Distribution")
    fig, ax = plt.subplots()
    df["Ticket Status"].value_counts().plot(kind="bar", ax=ax, color="skyblue")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Chart 2: Priority Distribution
with col2:
    st.markdown("### Ticket Priority Distribution")
    fig, ax = plt.subplots()
    df["Ticket Priority"].value_counts().plot(kind="bar", ax=ax, color="lightgreen")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Chart 3: Avg Resolution Time by Channel
with col1:
    st.markdown("### Avg Resolution Time by Channel")
    if "Resolution Time (hours)" in df.columns:
        fig, ax = plt.subplots()
        df.groupby("Ticket Channel")["Resolution Time (hours)"].mean().plot(kind="bar", ax=ax, color="salmon")
        ax.set_ylabel("Avg Hours")
        st.pyplot(fig)

# Chart 4: Avg Satisfaction by Priority
with col2:
    st.markdown("### Avg Satisfaction by Priority")
    if "Customer Satisfaction (1–5)" in df.columns:
        fig, ax = plt.subplots()
        df.groupby("Ticket Priority")["Customer Satisfaction (1–5)"].mean().plot(kind="bar", ax=ax, color="orange")
        ax.set_ylabel("Avg Rating")
        st.pyplot(fig)

# ===============================
# 📌 4. Correlation Heatmap
# ===============================
st.markdown("## 🔥 Correlation Heatmap")

# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

if not numeric_df.empty and numeric_df.shape[1] > 1:
    corr = numeric_df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.warning("Not enough numeric data to compute correlations.")

# ===============================
# 📌 5. Instructions & Notes
# ===============================
st.markdown("""
---
### 📝 Notes:
- All numeric fields are automatically cleaned (non-numeric values converted to NaN).
- Use the dropdowns above to filter tickets interactively.
- Correlation heatmap is shown only for valid numeric columns.
- Add more widgets or charts as needed!
""")
