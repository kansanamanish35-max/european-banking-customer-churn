import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Engagement Analytics", layout="wide")
st.title("Customer Engagement & Product Utilization Analytics")
st.caption("Retention strategy dashboard")

REQUIRED = ["CustomerId","Surname","CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited"]

@st.cache_data
def load(upload):
    return pd.read_csv(upload if upload is not None else "European_Bank.csv")

upload = st.sidebar.file_uploader("Upload customer CSV (optional)", type="csv")
try:
    df = load(upload)
except FileNotFoundError:
    st.error("Upload your CSV or place European_Bank.csv beside app.py.")
    st.stop()

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()

for c in ["Balance","EstimatedSalary","NumOfProducts","Exited","IsActiveMember","HasCrCard"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["Balance","EstimatedSalary","NumOfProducts","Exited","IsActiveMember"])
df["Exited"] = df["Exited"].astype(int)
df["IsActiveMember"] = df["IsActiveMember"].astype(int)
df["Product Depth Index"] = (df["NumOfProducts"] / max(df["NumOfProducts"].max(),1) * 100).clip(0,100)
high_balance_cut = df["Balance"].quantile(0.75)

df["Engagement Profile"] = np.select(
    [
        (df["IsActiveMember"].eq(1) & df["NumOfProducts"].ge(2)),
        (df["IsActiveMember"].eq(1) & df["NumOfProducts"].lt(2)),
        (df["IsActiveMember"].eq(0) & df["Balance"].ge(high_balance_cut)),
        (df["IsActiveMember"].eq(0) & df["Balance"].lt(high_balance_cut)),
    ],
    [
        "Active & Multi-Product",
        "Active & Low-Product",
        "Inactive High-Balance",
        "Inactive Other",
    ],
    default="Needs Review"
)
  
depth = df.NumOfProducts.map({1:25,2:70,3:100,4:100}).fillna(25)
df["Relationship Strength Index"] = (.40*df.IsActiveMember*100 + .35*depth + .25*df.HasCrCard.fillna(0)*100).round(1)

with st.sidebar:
    profiles = sorted(df["Engagement Profile"].unique())
    chosen = st.multiselect("Engagement profiles", profiles, default=profiles)
    lo, hi = int(df.NumOfProducts.min()), int(df.NumOfProducts.max())
    product_range = st.slider("Product count", lo, hi, (lo,hi))
    bal_cut = st.slider("High-balance threshold", 0.0, float(max(df.Balance.max(),1)), float(df.Balance.quantile(.75)))
    sal_cut = st.slider("Salary threshold", 0.0, float(max(df.EstimatedSalary.max(),1)), float(df.EstimatedSalary.quantile(.75)))

data = df[df["Engagement Profile"].isin(chosen) & df.NumOfProducts.between(*product_range)]
if data.empty:
    st.warning("No customers match these filters.")
    st.stop()

active = data[data.IsActiveMember.eq(1)]
inactive = data[data.IsActiveMember.eq(0)]
active_ret = 1-active.Exited.mean() if len(active) else np.nan
inactive_ret = 1-inactive.Exited.mean() if len(inactive) else np.nan
ratio = active_ret/inactive_ret if pd.notna(active_ret) and inactive_ret>0 else np.nan
a,b,c,d = st.columns(4)
a.metric("Customers", f"{len(data):,}")
b.metric("Churn rate", f"{data.Exited.mean()*100:.2f}%")
c.metric("Active churn", f"{active.Exited.mean()*100:.2f}%" if len(active) else "N/A")
d.metric("Inactive churn", f"{inactive.Exited.mean()*100:.2f}%" if len(inactive) else "N/A")

t1,t2,t3,t4 = st.tabs(["Engagement vs Churn","Product Utilization","High-Value Disengaged","Retention Strength"])
with t1:
    st.subheader("Churn by engagement profile")
    stats=data.groupby("Engagement Profile",observed=True).Exited.agg(Customers="size",Churn="mean").reset_index()
    stats["Churn (%)"]=stats.Churn*100
    fig,ax=plt.subplots()
    ax.bar(stats["Engagement Profile"],stats["Churn (%)"])
    ax.set_ylabel("Churn (%)"); ax.tick_params(axis="x",rotation=20); fig.tight_layout()
    st.pyplot(fig); plt.close(fig)
    st.metric("Engagement Retention Ratio",f"{ratio:.3f}" if pd.notna(ratio) else "N/A")
    st.caption("Active-member retention rate divided by inactive-member retention rate; descriptive association only.")
    st.write(stats[["Engagement Profile","Customers","Churn (%)"]].round(2).to_html(index=False),unsafe_allow_html=True)

with t2:
    st.subheader("Product utilization impact")
    ps=data.groupby("NumOfProducts",observed=True).Exited.agg(Customers="size",Churn="mean").reset_index()
    ps["Churn (%)"]=ps.Churn*100
    fig,ax=plt.subplots(); ax.bar(ps.NumOfProducts.astype(str),ps["Churn (%)"])
    ax.set_xlabel("Number of products"); ax.set_ylabel("Churn (%)"); fig.tight_layout()
    st.pyplot(fig); plt.close(fig)
    st.write(ps[["NumOfProducts","Customers","Churn (%)"]].round(2).to_html(index=False),unsafe_allow_html=True)
    st.metric("Mean Product Depth Index",f"{data['Product Depth Index'].mean():.1f}/100")
    st.caption("Product count divided by maximum count in the dataset, scaled to 100; a descriptive index, not a validated relationship measure.")

with t3:
    st.subheader("High-value disengaged customer detector")
    hb=data.Balance.ge(bal_cut)
    risk=data[hb & data.IsActiveMember.eq(0)]
    denom=int(hb.sum())
    disengage=len(risk)/denom*100 if denom else 0
    x,y,z=st.columns(3)
    x.metric("High-Balance Disengagement Rate",f"{disengage:.2f}%")
    y.metric("At-risk premium customers",f"{len(risk):,}")
    z.metric("At-risk churn",f"{risk.Exited.mean()*100:.2f}%" if len(risk) else "N/A")
    if len(risk):
        st.write(risk[["CustomerId","Geography","Age","Balance","EstimatedSalary","NumOfProducts","HasCrCard","Exited"]].sort_values("Balance",ascending=False).head(100).round(2).to_html(index=False),unsafe_allow_html=True)
    mismatch=data[data.Balance.ge(bal_cut)&data.EstimatedSalary.lt(sal_cut)]
    st.write(f"Salary–balance mismatch review: {len(mismatch):,} customers meet the selected balance/salary heuristic. This is not proof of financial stress.")

with t4:
    st.subheader("Retention strength scoring")
    st.caption("Rule-based framework: 40% activity + 35% product-depth tier (1=25, 2=70, 3+=100) + 25% card ownership. Not a trained or validated prediction.")
    score=data["Relationship Strength Index"]
    x,y,z=st.columns(3)
    x.metric("Mean Relationship Strength",f"{score.mean():.1f}/100")
    y.metric("Customers scoring ≥70",f"{(score>=70).sum():,}")
    z.metric("Churn in ≥70 group",f"{data.loc[score>=70,'Exited'].mean()*100:.2f}%" if (score>=70).any() else "N/A")
    card=data[data.HasCrCard.eq(1)]; nocard=data[data.HasCrCard.eq(0)]
    cr=1-card.Exited.mean() if len(card) else np.nan
    nr=1-nocard.Exited.mean() if len(nocard) else np.nan
    stick=cr/nr if pd.notna(cr) and pd.notna(nr) and nr>0 else np.nan
    st.metric("Credit Card Stickiness Score",f"{stick:.3f}" if pd.notna(stick) else "N/A")
    st.caption("Card-holder retention divided by non-holder retention; association does not establish causation.")

st.divider()
st.subheader("Responsible interpretation")
st.markdown("- Use flagged customers as a human-reviewed outreach queue, not an automated decision.\n- These are associations, not proof of causation.\n- Product groups with few customers can have unstable churn rates.\n- Activity is proxied by the available active-member field; transaction-level engagement data is not provided.")
