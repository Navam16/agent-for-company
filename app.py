# ======================================
# app.py — AI Business Analyst Agent
# ======================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq

# --------------------------------------
# PAGE CONFIG
# --------------------------------------
st.set_page_config(
    page_title="AI Business Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Business Analyst Agent")
st.caption("Ask business questions in natural language")

# --------------------------------------
# LOAD GROQ CLIENT (Secrets)
# --------------------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --------------------------------------
# LOAD DATA (FILES IN REPO ROOT)
# --------------------------------------
@st.cache_data
def load_data():
    return {
        "Online Sales": pd.read_csv("Online_Sales.csv"),
        "Discount Coupon": pd.read_csv("Discount_Coupon.csv"),
        "Marketing Spend": pd.read_csv("Marketing_Spend.csv"),
        "Customer Data": pd.read_excel("CustomersData.xlsx"),
        "Tax Amount": pd.read_excel("Tax_amount.xlsx")
    }

datasets = load_data()

online_sales = datasets["Online Sales"]
discount_coupon = datasets["Discount Coupon"]
marketing_spend = datasets["Marketing Spend"]
customer_data = datasets["Customer Data"]
tax_amount = datasets["Tax Amount"]

# --------------------------------------
# SIDEBAR — DATASET SELECTION ONLY
# --------------------------------------
st.sidebar.title("📁 Dataset Selection")

selected_dataset_name = st.sidebar.selectbox(
    "Choose a dataset",
    list(datasets.keys())
)

# --------------------------------------
# MAIN DASHBOARD — DATASET PREVIEW
# --------------------------------------
st.markdown("## 🔍 Dataset Preview")

st.write(
    f"**{selected_dataset_name}** — "
    f"{datasets[selected_dataset_name].shape[0]} rows × "
    f"{datasets[selected_dataset_name].shape[1]} columns"
)

st.dataframe(
    datasets[selected_dataset_name].head(20),
    use_container_width=True
)

st.divider()

# --------------------------------------
# DATA PREPROCESSING
# --------------------------------------
online_sales["Transaction_Date"] = pd.to_datetime(
    online_sales["Transaction_Date"], errors="coerce"
)
online_sales["Revenue"] = online_sales["Quantity"] * online_sales["Avg_Price"]
online_sales["Year"] = online_sales["Transaction_Date"].dt.year
online_sales["Month"] = online_sales["Transaction_Date"].dt.month
online_sales["Coupon_Status"] = online_sales["Coupon_Status"].fillna("No Coupon")

discount_coupon["Month"] = discount_coupon["Month"].astype(str)
discount_coupon["Discount_pct"] = discount_coupon["Discount_pct"].fillna(0)

marketing_spend["Date"] = pd.to_datetime(marketing_spend["Date"], errors="coerce")
marketing_spend["Year"] = marketing_spend["Date"].dt.year
marketing_spend["Month"] = marketing_spend["Date"].dt.month
marketing_spend["Total_Marketing_Spend"] = (
    marketing_spend["Offline_Spend"] + marketing_spend["Online_Spend"]
)

tax_amount["GST"] = tax_amount["GST"].fillna(0)

# --------------------------------------
# LOOKUPS
# --------------------------------------
tax_lookup = tax_amount.set_index("Product_Category")["GST"]

discount_lookup = discount_coupon.set_index(
    ["Month", "Product_Category"]
)["Discount_pct"]

marketing_lookup = marketing_spend.groupby(
    ["Year", "Month"]
)["Total_Marketing_Spend"].sum()

online_sales["GST_pct"] = online_sales["Product_Category"].map(tax_lookup)

online_sales["Discount_pct"] = (
    online_sales
    .set_index(["Month", "Product_Category"])
    .index.map(discount_lookup)
    .fillna(0)
)

online_sales["Marketing_Spend"] = (
    online_sales
    .set_index(["Year", "Month"])
    .index.map(marketing_lookup)
)

# --------------------------------------
# ANALYSIS FUNCTIONS
# --------------------------------------
def sales_trend(df):
    return (
        df.groupby(["Year", "Month", "Product_Category"])["Revenue"]
        .sum()
        .reset_index()
    )

def underperforming_products(df):
    return (
        df.groupby("Product_Category")
        .agg(
            total_revenue=("Revenue", "sum"),
            avg_discount=("Discount_pct", "mean"),
            total_quantity=("Quantity", "sum")
        )
        .reset_index()
        .sort_values("total_revenue")
    )

def discount_vs_revenue(df):
    return (
        df.groupby("Product_Category")
        .agg(
            avg_discount_pct=("Discount_pct", "mean"),
            total_revenue=("Revenue", "sum")
        )
        .reset_index()
    )

INTENT_TO_ANALYSIS = {
    "sales_trend": sales_trend,
    "underperforming_products": underperforming_products,
    "discount_vs_revenue": discount_vs_revenue
}

# --------------------------------------
# INTENT CLASSIFICATION (Groq)
# --------------------------------------
def classify_intent(query):
    prompt = f"""
Classify into ONE intent:
- sales_trend
- underperforming_products
- discount_vs_revenue
- unknown

Return ONLY the intent name.
Question: {query}
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    intent = res.choices[0].message.content.strip().lower()
    return intent if intent in INTENT_TO_ANALYSIS else "unknown"

# --------------------------------------
# EXPLANATION
# --------------------------------------
def explain_result(query, df_preview):
    prompt = f"""
You are a business analyst.

Question:
{query}

Analysis output (sample):
{df_preview}

Explain:
1. What is happening
2. Why it might be happening
3. Business implication
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return res.choices[0].message.content

# --------------------------------------
# EXAMPLE QUESTIONS (NOW BELOW DATA)
# --------------------------------------
st.markdown("## 💡 Example Questions You Can Ask")

example_queries = [
    "Which product categories generate high revenue with relatively low marketing spend?",
    "How do discounts impact revenue differently across product categories?",
    "Which products perform well in California but underperform in New York?",
    "Are there categories where offline marketing is more effective than online?",
    "Which products depend heavily on discounts to drive volume?",
    "How does marketing spend correlate with sales for Nest and Google products?",
    "Which categories have high sales volume but low overall revenue?",
    "Where should marketing budgets be reallocated for better ROI?",
    "Which products are consistently underperforming despite promotions?",
    "Based on all data, what should be the top business focus areas?"
]

selected_query = st.radio(
    "👇 Click a question or ask your own:",
    example_queries,
    index=None
)

st.divider()

# --------------------------------------
# USER INPUT
# --------------------------------------
query = st.text_input(
    "🧠 Ask a business question",
    value=selected_query if selected_query else "",
    placeholder="Type or select a question above"
)

# --------------------------------------
# RUN AGENT
# --------------------------------------
if query:
    intent = classify_intent(query)
    st.markdown(f"### 🎯 Intent: `{intent}`")

    if intent == "unknown":
        st.warning("I couldn’t clearly understand this question.")
    else:
        result_df = INTENT_TO_ANALYSIS[intent](online_sales)

        st.markdown("### 📊 Analysis Preview")
        st.dataframe(result_df.head(20), use_container_width=True)

        st.markdown("### 📝 Explanation")
        explanation = explain_result(query, result_df.head())
        st.write(explanation)
