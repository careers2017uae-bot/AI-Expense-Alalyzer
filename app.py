import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import date
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize client
client = OpenAI(api_key=openai_api_key)

# CSV file for local data storage
DATA_FILE = "expenses.csv"

# App title
st.set_page_config(page_title="💰 AI Expense Analyzer", layout="wide")
st.title("💰 AI Expense Analyzer")
st.write("Track, analyze, and get smart recommendations for your expenses using AI!")

# Initialize data
if not os.path.exists(DATA_FILE):
    df = pd.DataFrame(columns=["Date", "Amount", "Description", "Category", "LLM_Category", "Confidence"])
    df.to_csv(DATA_FILE, index=False)
else:
    df = pd.read_csv(DATA_FILE)

# Sidebar: Expense Entry
st.sidebar.header("➕ Add Expense")
with st.sidebar.form("expense_form"):
    date_input = st.date_input("Date", value=date.today())
    amount = st.number_input("Amount ($)", min_value=0.0, step=0.01)
    description = st.text_input("Description")
    category = st.selectbox(
        "Category (optional)",
        ["", "Food & Drink", "Shopping", "Bills", "Transport", "Entertainment", "Others"]
    )
    submit = st.form_submit_button("Add & Analyze")

# Function: Call LLM for expense categorization
def analyze_with_llm(description, amount):
    if not openai_api_key:
        return "Unknown", 0.0
    prompt = f"""
    You are an expert expense analyzer.
    Categorize this expense based on description and amount.
    Categories: Food & Drink, Shopping, Bills, Transport, Entertainment, Others.

    Return JSON with keys:
    - category
    - confidence (0.00-1.00)
    
    Expense description: {description}
    Amount: {amount}
    """
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0.2
        )
        import json
        text = response.output_text.strip()
        data = json.loads(text)
        return data.get("category", "Unknown"), data.get("confidence", 0.0)
    except Exception as e:
        st.error(f"LLM error: {e}")
        return "Unknown", 0.0

# On form submission
if submit:
    if description and amount > 0:
        llm_cat, conf = analyze_with_llm(description, amount)
        new_entry = {
            "Date": date_input,
            "Amount": amount,
            "Description": description,
            "Category": category if category else None,
            "LLM_Category": llm_cat,
            "Confidence": round(conf, 2)
        }
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        st.sidebar.success(f"Expense added ✅ — LLM suggests: {llm_cat} ({conf*100:.0f}% confident)")
    else:
        st.sidebar.warning("Please fill all required fields.")

# Display expenses
st.subheader("📊 Expense Records")
st.dataframe(df, use_container_width=True)

# Charts Section
if not df.empty:
    col1, col2 = st.columns(2)

    # Bar chart - Amount by LLM_Category
    with col1:
        fig_bar = px.bar(df, x="LLM_Category", y="Amount", color="LLM_Category",
                         title="Expenses by LLM Category", text_auto=True)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Pie chart - Share by LLM_Category
    with col2:
        fig_pie = px.pie(df, values="Amount", names="LLM_Category", title="Expense Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Line chart - Expense over Time
    st.subheader("📈 Spending Trend")
    fig_line = px.line(df, x="Date", y="Amount", markers=True, title="Expense Over Time")
    st.plotly_chart(fig_line, use_container_width=True)

# AI summary
if not df.empty:
    st.divider()
    st.subheader("🤖 AI Recommendations")

    expenses_summary = df.groupby("LLM_Category")["Amount"].sum().to_dict()
    prompt_summary = f"""
    You are a financial advisor.
    Analyze the following spending summary and give 3 concise recommendations (≤40 words each) 
    to help the user manage their expenses better.
    {expenses_summary}
    """
    if st.button("Generate AI Recommendations"):
        try:
            response = client.responses.create(model="gpt-4.1-mini", input=prompt_summary, temperature=0.4)
            st.success("Here are your AI recommendations:")
            st.write(response.output_text)
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
