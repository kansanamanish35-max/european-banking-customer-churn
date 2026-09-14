import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="European Banking Churn Analytics",
    page_icon="🏦",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("European_Bank.csv")
    return df

df = load_data()

# -----------------------------
# Derived segmentation columns
# -----------------------------
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 29, 45, 60, np.inf],
    labels=["<30", "30–45", "46–60", "60+"]
)

df["CreditScoreGroup"] = pd.cut(
    df["CreditScore"],
    bins=[0, 579, 669, np.inf],
    labels=["Low", "Medium", "High"]
)

df["TenureGroup"] = pd.cut(
    df["Tenure"],
    bins=[-1, 2, 5, 10],
    labels=["New (0–2)", "Mid (3–5)", "Long (6–10)"]
)

q75 = df["Balance"].quantile(0.75)
df["HighValue"] = np.where(df["Balance"] >= q75, "High Value", "Regular")
df["ActivityGroup"] = np.where(
    df["IsActiveMember"] == 1, "Active", "Inactive"
)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

geographies = st.sidebar.multiselect(
    "Geography",
    options=sorted(df["Geography"].dropna().unique()),
    default=sorted(df["Geography"].dropna().unique())
)

genders = st.sidebar.multiselect(
    "Gender",
    options=sorted(df["Gender"].dropna().unique()),
    default=sorted(df["Gender"].dropna().unique())
)

age_groups = st.sidebar.multiselect(
    "Age Group",
    options=df["AgeGroup"].cat.categories.tolist(),
    default=df["AgeGroup"].cat.categories.tolist()
)

activities = st.sidebar.multiselect(
    "Activity",
    options=["Active", "Inactive"],
    default=["Active", "Inactive"]
)

filtered = df[
    df["Geography"].isin(geographies)
    & df["Gender"].isin(genders)
    & df["AgeGroup"].astype(str).isin(age_groups)
    & df["ActivityGroup"].isin(activities)
].copy()

# -----------------------------
# Header
# -----------------------------
st.title("🏦 Customer Segmentation & Churn Pattern Analytics")
st.caption("European Banking | Internship Project")

if filtered.empty:
    st.warning("No customers match the selected filters.")
    st.stop()

# -----------------------------
# KPI cards
# -----------------------------
total = len(filtered)
churned = int(filtered["Exited"].sum())
retained = total - churned
churn_rate = churned / total * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{total:,}")
c2.metric("Churned", f"{churned:,}")
c3.metric("Retained", f"{retained:,}")
c4.metric("Churn Rate", f"{churn_rate:.2f}%")

st.divider()

# -----------------------------
# Geography
# -----------------------------
st.subheader("Geography-wise Churn")

geo = (
    filtered.groupby("Geography", observed=False)["Exited"]
    .agg(["count", "sum"])
    .reset_index()
)
geo["Churn Rate"] = geo["sum"] / geo["count"] * 100

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(data=geo, x="Geography", y="Churn Rate", ax=ax)
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("")
ax.set_title("Churn Rate by Geography")
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%")
st.pyplot(fig)
plt.close(fig)

# -----------------------------
# Age and gender
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Age Group Churn")
    age = (
        filtered.groupby("AgeGroup", observed=False)["Exited"]
        .agg(["count", "sum"])
        .reset_index()
    )
    age["Churn Rate"] = age["sum"] / age["count"] * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=age, x="AgeGroup", y="Churn Rate", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("")
    ax.set_title("Churn by Age Group")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Gender Churn")
    gender = (
        filtered.groupby("Gender")["Exited"]
        .agg(["count", "sum"])
        .reset_index()
    )
    gender["Churn Rate"] = gender["sum"] / gender["count"] * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=gender, x="Gender", y="Churn Rate", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("")
    ax.set_title("Churn by Gender")
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# Activity and products
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Activity Churn")
    act = (
        filtered.groupby("ActivityGroup")["Exited"]
        .agg(["count", "sum"])
        .reset_index()
    )
    act["Churn Rate"] = act["sum"] / act["count"] * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=act, x="ActivityGroup", y="Churn Rate", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("")
    ax.set_title("Active vs Inactive Customers")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Product Usage")
    prod = (
        filtered.groupby("NumOfProducts")["Exited"]
        .agg(["count", "sum"])
        .reset_index()
    )
    prod["Churn Rate"] = prod["sum"] / prod["count"] * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=prod, x="NumOfProducts", y="Churn Rate", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Number of Products")
    ax.set_title("Churn by Number of Products")
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# Credit score and tenure
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Credit Score")
    credit = (
        filtered.groupby("CreditScoreGroup", observed=False)["Exited"]
        .agg(["count", "sum"])
        .reset_index()
    )
    credit["Churn Rate"] = credit["sum"] / credit["count"] * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=credit, x="CreditScoreGroup", y="Churn Rate", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("")
    ax.set_title("Churn by Credit Score Group")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Tenure")
    tenure = (
        filtered.groupby("TenureGroup", observed=False)["Exited"]
        .agg(["count", "sum"])
        .reset_index()
    )
    tenure["Churn Rate"] = tenure["sum"] / tenure["count"] * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=tenure, x="TenureGroup", y="Churn Rate", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("")
    ax.set_title("Churn by Tenure Group")
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# Balance / high-value
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Balance Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(filtered["Balance"], bins=30, kde=True, ax=ax)
    ax.set_title("Customer Balance Distribution")
    ax.set_xlabel("Balance")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("High-Value Customers")
    hv = (
        filtered.groupby("HighValue")["Exited"]
        .agg(["count", "sum"])
        .reset_index()
    )
    hv["Churn Rate"] = hv["sum"] / hv["count"] * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=hv, x="HighValue", y="Churn Rate", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("")
    ax.set_title(f"High-Value Threshold: {q75:,.2f}")
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# Customer explorer
# -----------------------------
st.subheader("Customer Explorer")

display_cols = [
    "CustomerId", "Surname", "CreditScore", "Geography", "Gender",
    "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
]
display_cols = [c for c in display_cols if c in filtered.columns]

st.markdown(
    filtered[display_cols].head(200).to_html(index=False),
    unsafe_allow_html=True
)

st.caption(
    "Internship Project | Customer Segmentation & Churn Pattern Analytics in European Banking"
)
