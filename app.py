import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import date
from dotenv import load_dotenv
from groq import Groq
import json

# -----------------------------
# ✅ Setup and Configuration
# -----------------------------
st.set_page_config(page_title="💰 AI Expense Analyzer (Groq)", layout="wide")
st.title("💰 AI Expense Analyzer")
st.write("Track, analyze, and get smart recommendations for your expenses using Groq LLM!")

# Load environment variables (optional)
load_dotenv()

# If you want to store key in .env, add: GROQ_API_KEY=yourkey
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ Groq API key not found! Please check your .env file.")
else:
    st.success("✅ Groq API key loaded successfully.")
# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

DATA_FILE = "expenses.csv"

# -----------------------------
# ✅ Initialize Data
# -----------------------------
if not os.path.exists(DATA_FILE):
    df = pd.DataFrame(columns=["Date", "Amount", "Description", "Category", "LLM_Category", "Confidence"])
    df.to_csv(DATA_FILE, index=False)
else:
    df = pd.read_csv(DATA_FILE)

# -----------------------------
# 🧾 Sidebar: Expense Entry
# -----------------------------
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

# -----------------------------
# 🤖 Groq-based LLM Categorization
# -----------------------------
def analyze_with_llm(description, amount):
    prompt = f"""
    You are an expert financial assistant. Categorize this expense based on the description and amount.
    Possible categories: Food & Drink, Shopping, Bills, Transport, Entertainment, Others.

    Return a JSON object only with:
    - category
    - confidence (0.0–1.0)

    Example output:
    {{ "category": "Food & Drink", "confidence": 0.95 }}

    Expense description: {description}
    Amount: {amount}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = response.choices[0].message.content.strip()
        data = json.loads(text)
        return data.get("category", "Unknown"), data.get("confidence", 0.0)
    except Exception as e:
        st.error(f"LLM error: {e}")
        return "Unknown", 0.0

# -----------------------------
# 📥 On Form Submission
# -----------------------------
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

# -----------------------------
# 📊 Expense Table
# -----------------------------
st.subheader("📋 Expense Records")
st.dataframe(df, use_container_width=True)

# -----------------------------
# 📈 Charts
# -----------------------------
if not df.empty:
    col1, col2 = st.columns(2)

    # Bar Chart - Expenses by LLM Category
    with col1:
        fig_bar = px.bar(df, x="LLM_Category", y="Amount", color="LLM_Category",
                         title="Expenses by LLM Category", text_auto=True)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Pie Chart - Expense Distribution
    with col2:
        fig_pie = px.pie(df, values="Amount", names="LLM_Category", title="Expense Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Line Chart - Trend Over Time
    st.subheader("📈 Spending Trend")
    fig_line = px.line(df, x="Date", y="Amount", markers=True, title="Expenses Over Time")
    st.plotly_chart(fig_line, use_container_width=True)

# -----------------------------
# 🧠 AI Recommendations
# -----------------------------
if not df.empty:
    st.divider()
    st.subheader("🤖 AI Recommendations")

    # Summarize user spending for Groq
    summary = df.groupby("LLM_Category")["Amount"].sum().to_dict()
    prompt_summary = f"""
    You are a helpful financial assistant.
    The user's total spending by category is:
    {summary}

    Please generate 3 short, practical recommendations (each under 40 words) 
    to help the user manage their spending more effectively.
    """

    if st.button("Generate AI Recommendations"):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt_summary}],
                temperature=0.4,
            )
            st.success("Here are your AI recommendations:")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
