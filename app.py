import streamlit as st
import pandas as pd

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="📊 AI Business Analyst Agent",
    layout="wide"
)

st.title("📊 AI Business Analyst Agent")
st.caption("Ask business questions in natural language")

# -------------------------------
# Data loading
# -------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    try:
        online_sales = pd.read_csv("Online_Sales.csv")
        discount_coupon = pd.read_csv("Discount_Coupon.csv")
        marketing_spend = pd.read_csv("Marketing_Spend.csv")
        customer_data = pd.read_excel("CustomersData.xlsx")
        tax_amount = pd.read_excel("Tax_amount.xlsx")

        return online_sales, discount_coupon, marketing_spend, customer_data, tax_amount

    except Exception as e:
        st.error("❌ Error loading datasets. Please check file names and formats.")
        st.exception(e)
        st.stop()


online_sales, discount_coupon, marketing_spend, customer_data, tax_amount = load_data()

# -------------------------------
# Sidebar info
# -------------------------------
st.sidebar.success("✅ Data loaded successfully")

st.sidebar.markdown("### 📁 Dataset Overview")
st.sidebar.write(f"Online Sales rows: {online_sales.shape[0]}")
st.sidebar.write(f"Discount Coupon rows: {discount_coupon.shape[0]}")
st.sidebar.write(f"Marketing Spend rows: {marketing_spend.shape[0]}")
st.sidebar.write(f"Customer Data rows: {customer_data.shape[0]}")
st.sidebar.write(f"Tax Amount rows: {tax_amount.shape[0]}")

# -------------------------------
# Dataset preview
# -------------------------------
with st.expander("🔍 Preview Datasets"):
    st.subheader("Online Sales")
    st.dataframe(online_sales.head())

    st.subheader("Discount Coupon")
    st.dataframe(discount_coupon.head())

    st.subheader("Marketing Spend")
    st.dataframe(marketing_spend.head())

    st.subheader("Customer Data")
    st.dataframe(customer_data.head())

    st.subheader("Tax Amount")
    st.dataframe(tax_amount.head())

# -------------------------------
# Business Question Input
# -------------------------------
st.markdown("## 💬 Ask a Business Question")

example_queries = [
    "Which product categories show high sales but low marketing spend?",
    "How do discounts impact repeat customers across regions?",
    "Identify months where tax impact reduced net revenue significantly",
    "Which customer segment is most sensitive to discount coupons?",
    "Compare online sales growth with marketing spend efficiency"
]

query = st.text_input(
    "Type your business question here:",
    placeholder="e.g. Which region has the highest profit margin after tax?"
)

st.markdown("**Try these examples:**")
for q in example_queries:
    st.markdown(f"- {q}")

# -------------------------------
# Placeholder response logic
# -------------------------------
if query:
    st.markdown("## 📈 Analysis Result")

    st.info(
        "🧠 This is where your AI / agent / LLM logic will plug in.\n\n"
        "Right now, the app confirms:\n"
        "- Data is accessible\n"
        "- Queries are captured\n"
        "- Streamlit Cloud deployment is stable"
    )

    st.write("**Your question:**")
    st.write(query)

    st.success("✅ App is running correctly. Ready for AI integration.")
