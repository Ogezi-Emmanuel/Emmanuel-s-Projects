import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from datetime import datetime

# Function to get dynamic month names
def get_month_names(num_months=6):
    today = datetime.now()
    months = []
    for i in range(num_months):
        month_date = datetime(today.year, today.month, 1) - pd.DateOffset(months=i)
        months.append(month_date.strftime("%B"))
    return months[::-1] # Return in chronological order (oldest to newest)

# Get dynamic month names
month_names = get_month_names()

# Function to load CSS

# Load the custom CSS
with open(os.path.join(os.path.dirname(__file__), "..", "styles", "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Page Configuration ---
st.set_page_config(page_title="Credit Scoring", page_icon="📊", layout="wide")

st.title("Predictive Credit Scoring (Credit Card Default)")

st.markdown("""
This project aims to predict customer churn for a digital bank, helping to identify at-risk customers proactively. The models were trained on a dataset of 30,000 rows with the following features:

- CustomerId: Unique identifier for each customer.
- Tenure: Number of months the customer has been with the bank.
- Balance: Current account balance.
- LoginFrequency_LastMonth: Number of times the customer logged in during the last month.
- Transactions_LastMonth: Number of transactions made in the last month.
- ServiceCalls_Last3Months: Number of service calls made in the last 3 months.
- UsedSavingsFeature: Whether the customer used a savings feature (0 = No, 1 = Yes).
""")

# Define input features based on the provided list, selecting the most important ones
# For simplicity, I'm making an initial selection. We can refine this based on your input.
# You mentioned: LIMIT_BAL SEX EDUCATION MARRIAGE AGE PAY_1 PAY_2 PAY_3 PAY_4 PAY_5 ... BILL_AMT4 BILL_AMT5 BILL_AMT5 BILL_AMT6 PAY_AMT1 PAY_AMT2 PAY_AMT3 PAY_AMT4 PAY_AMT5 PAY_AMT6 default_payment_next_month
# Let's pick a representative set for the UI.

limit_bal = st.number_input("Limit Balance", min_value=0, value=50000)
sex = st.radio("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female") # 1=male, 2=female
education = st.selectbox("Education", options=[1, 2, 3, 4], format_func=lambda x: {1: "Graduate School", 2: "University", 3: "High School", 4: "Other"}[x]) # 1=graduate school, 2=university, 3: "High School", 4: "Other"}[x]) # 1=graduate school, 2=university, 3=high school, 4=other
marriage = st.selectbox("Marriage", options=[1, 2, 3], format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x]) # 1=married, 2=single, 3=other
age = st.slider("Age", min_value=18, max_value=70, value=30)

# Payment status
with st.expander("Repayment Status"):
    pay_options = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    pay_format_func = lambda x: {
        -2: "No consumption", -1: "Pay duly", 0: "Use revolving credit",
        1: "Delay for one month", 2: "Delay for two months", 3: "Delay for three months",
        4: "Delay for four months", 5: "Delay for five months", 6: "Delay for six months",
        7: "Delay for seven months", 8: "Delay for eight months", 9: "Delay for nine months and above"
    }[x]
    col1, col2, col3 = st.columns(3)
    with col1:
        pay_1 = st.selectbox(f"Repayment Status ({month_names[-1]})", options=pay_options, format_func=pay_format_func)
        pay_2 = st.selectbox(f"Repayment Status ({month_names[-2]})", options=pay_options, format_func=pay_format_func)
    with col2:
        pay_3 = st.selectbox(f"Repayment Status ({month_names[-3]})", options=pay_options, format_func=pay_format_func)
        pay_4 = st.selectbox(f"Repayment Status ({month_names[-4]})", options=pay_options, format_func=pay_format_func)
    with col3:
        pay_5 = st.selectbox(f"Repayment Status ({month_names[-5]})", options=pay_options, format_func=pay_format_func)
        pay_6 = st.selectbox(f"Repayment Status ({month_names[-6]})", options=pay_options, format_func=pay_format_func)

# Bill amount
with st.expander("Bill Statement Amount"):
    col1, col2, col3 = st.columns(3)
    with col1:
        bill_amt1 = st.number_input(f"Bill Statement ({month_names[-1]})", min_value=0, value=10000)
        bill_amt2 = st.number_input(f"Bill Statement ({month_names[-2]})", min_value=0, value=10000)
    with col2:
        bill_amt3 = st.number_input(f"Bill Statement ({month_names[-3]})", min_value=0, value=10000)
        bill_amt4 = st.number_input(f"Bill Statement ({month_names[-4]})", min_value=0, value=10000)
    with col3:
        bill_amt5 = st.number_input(f"Bill Statement ({month_names[-5]})", min_value=0, value=10000)
        bill_amt6 = st.number_input(f"Bill Statement ({month_names[-6]})", min_value=0, value=10000)

# Amount paid
with st.expander("Previous Payment Amount"):
    col1, col2, col3 = st.columns(3)
    with col1:
        pay_amt1 = st.number_input(f"Payment Amount ({month_names[-1]})", min_value=0, value=1000)
        pay_amt2 = st.number_input(f"Payment Amount ({month_names[-2]})", min_value=0, value=1000)
    with col2:
        pay_amt3 = st.number_input(f"Payment Amount ({month_names[-3]})", min_value=0, value=1000)
        pay_amt4 = st.number_input(f"Payment Amount ({month_names[-4]})", min_value=0, value=1000)
    with col3:
        pay_amt5 = st.number_input(f"Payment Amount ({month_names[-5]})", min_value=0, value=1000)
        pay_amt6 = st.number_input(f"Payment Amount ({month_names[-6]})", min_value=0, value=1000)

# Create a DataFrame for the input features
input_data = pd.DataFrame([[
    limit_bal, sex, education, marriage, age,
    pay_1, pay_2, pay_3, pay_4, pay_5, pay_6,
    bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
    pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6
]],
                          columns=[
                              'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                              'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                              'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                              'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
                          ])

# Model selection tabs
tab1, tab2 = st.tabs(["LightGBM Model", "Logistic Regression Model"])

with tab1:
    st.header("LightGBM Model Prediction")
    # Load the appropriate model
    @st.cache_resource
    def load_lgbm_model():
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "credit_scoring_lgbm_model.joblib"))
        try:
            model = joblib.load(model_path)
            return model
        except FileNotFoundError:
            st.error(f"Error: Model file not found for LightGBM. Please ensure '{model_path}' exists.")
            st.stop()
    lgbm_model = load_lgbm_model()
    
    # Make prediction
    if st.button("Predict Creditworthiness with LightGBM"):
        with st.spinner('Predicting creditworthiness...'):
            try:
                prediction = lgbm_model.predict(input_data)
                prediction_proba = lgbm_model.predict_proba(input_data)
    
                st.subheader("Prediction Result")
    
                # Display prediction with color coding and emojis
                if prediction[0] == 0:
                    st.success("✨ The person is likely to have **Good Credit**! ✨")
                else:
                    st.error("💔 The person is likely to have **Bad Credit**! 💔")
    
                st.write(f"Probability of Good Credit: {prediction_proba[0][0]:.2f}")
                st.write(f"Probability of Bad Credit: {prediction_proba[0][1]:.2f}")
    
                st.subheader("Credit Classification Chart")
    
                if prediction[0] == 0:
                    credit_class = "Good Credit"
                else:
                    if prediction_proba[0][1] < 0.4:
                        credit_class = "Average Credit"
                    else:
                        credit_class = "Bad Credit"
    
                if credit_class == "Good Credit":
                    st.markdown(f"<div class='credit-result good-credit'><h2>{credit_class}</h2></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='credit-result bad-credit'><h2>{credit_class}</h2></div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

with tab2:
    st.header("Logistic Regression Model Prediction")
    # Load the appropriate model
    @st.cache_resource
    def load_log_reg_model():
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "credit_scoring_log_reg_pipeline.joblib"))
        try:
            pipeline = joblib.load(model_path)
            return pipeline
        except FileNotFoundError:
            st.error(f"Error: Model file not found for Logistic Regression. Please ensure '{model_path}' exists.")
            st.stop()
    log_reg_model = load_log_reg_model()
    
    # Make prediction
    if st.button("Predict Creditworthiness with Logistic Regression"):
        with st.spinner('Predicting creditworthiness...'):
            try:
                prediction = log_reg_model.predict(input_data)
                prediction_proba = log_reg_model.predict_proba(input_data)
    
                st.subheader("Prediction Result")
    
                # Display prediction with color coding and emojis
                if prediction[0] == 0:
                    st.success("✨ The person is likely to have **Good Credit**! ✨")
                else:
                    st.error("💔 The person is likely to have **Bad Credit**! 💔")
    
                st.write(f"Probability of Good Credit: {prediction_proba[0][0]:.2f}")
                st.write(f"Probability of Bad Credit: {prediction_proba[0][1]:.2f}")
    
                st.subheader("Credit Classification Chart")
    
                if prediction[0] == 0:
                    credit_class = "Good Credit"
                else:
                    if prediction_proba[0][1] < 0.4:
                        credit_class = "Average Credit"
                    else:
                        credit_class = "Bad Credit"
    
                if credit_class == "Good Credit":
                    st.markdown(f"<div class='credit-result good-credit'><h2>{credit_class}</h2></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='credit-result bad-credit'><h2>{credit_class}</h2></div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")