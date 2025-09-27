import streamlit as st
import pandas as pd
import joblib

# Function to load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the custom CSS
load_css("c:/Users/user/Emmanuel's Projects/styles/style.css")

# Load the trained models
try:
    logistic_regression_model = joblib.load('models/churn_prediction_log_reg_model.joblib')
    random_forest_model = joblib.load('models/churn_prediction_random_forest_model.joblib')
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'churn_prediction_log_reg_model.joblib' and 'churn_prediction_random_forest_model.joblib' are in the 'models' directory.")
    st.stop()

st.set_page_config(page_title="Churn Prediction", layout="wide", page_icon="../images/logo.jpeg")

st.title("Proactive Churn Prediction for a Digital Bank")

st.markdown("""
This project aims to predict customer churn for a digital bank, helping to identify at-risk customers proactively.
The models were trained on a dataset of 40,000 rows with the following features:
- `CustomerId`: Unique identifier for each customer.
- `Tenure`: Number of months the customer has been with the bank.
- `Balance`: Current account balance.
- `LoginFrequency_LastMonth`: Number of times the customer logged in during the last month.
- `Transactions_LastMonth`: Number of transactions made in the last month.
- `ServiceCalls_Last3Months`: Number of service calls made in the last 3 months.
- `UsedSavingsFeature`: Whether the customer used a savings feature (0 = No, 1 = Yes).
""")

st.markdown("For a detailed technical analysis of how this model was trained, please visit the [GitHub Repository](https://github.com/Ogezi-Emmanuel/Proactive-Customer-Churn-Analysis).")

st.header("Customer Information Input")

with st.form("churn_prediction_form"):
    customer_id = st.text_input("Customer ID", "C12345")
    tenure = st.slider("Tenure (Months)", 0, 120, 24)
    balance = st.number_input("Balance", 0.0, 100000.0, 5000.0, step=100.0)
    login_frequency = st.slider("Login Frequency (Last Month)", 0, 30, 5)
    transactions_last_month = st.slider("Transactions (Last Month)", 0, 50, 10)
    service_calls_last_3_months = st.slider("Service Calls (Last 3 Months)", 0, 10, 1)
    used_savings_feature = st.selectbox("Used Savings Feature", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    submitted = st.form_submit_button("Predict Churn")

    if submitted:
        input_data = pd.DataFrame([[
            customer_id, tenure, balance, login_frequency,
            transactions_last_month, service_calls_last_3_months, used_savings_feature
        ]], columns=[
            'CustomerId', 'Tenure', 'Balance', 'LoginFrequency_LastMonth',
            'Transactions_LastMonth', 'ServiceCalls_Last3Months', 'UsedSavingsFeature'
        ])

        # Drop CustomerId for prediction as it's not a feature
        features_for_prediction = input_data.drop('CustomerId', axis=1)

        # Make predictions
        lr_prediction = logistic_regression_model.predict(features_for_prediction)[0]
        rf_prediction = random_forest_model.predict(features_for_prediction)[0]

        st.subheader("Prediction Results")
        st.write(f"**Customer ID:** {customer_id}")

        st.markdown("---")
        st.markdown("### Logistic Regression (High Recall) Prediction")
        if lr_prediction == 1:
            st.error("This customer is likely to churn.")
            st.write("This model is like casting a wide net. It successfully finds 93% of all customers who are going to churn, which is excellent for being comprehensive. However, its very low precision (46%) means that for every 100 customers it flags, 54 are actually happy customers who were not going to leave.")
            st.write("Use this model if: Your retention campaign is extremely cheap (e.g., an automated email) and the business goal is to contact every single potential churner, no matter the cost of contacting happy customers by mistake.")
        else:
            st.success("This customer is unlikely to churn.")
            st.write("This model is like casting a wide net. It successfully finds 93% of all customers who are going to churn, which is excellent for being comprehensive. However, its very low precision (46%) means that for every 100 customers it flags, 54 are actually happy customers who were not going to leave.")
            st.write("Use this model if: Your retention campaign is extremely cheap (e.g., an automated email) and the business goal is to contact every single potential churner, no matter the cost of contacting happy customers by mistake.")

        st.markdown("---")
        st.markdown("### Random Forest (Balanced Performance) Prediction")
        if rf_prediction == 1:
            st.error("This customer is likely to churn.")
            st.write("This model is more like a skilled sniper. It is much more precise. When it flags a customer as a churn risk, it's correct 71% of the time. The trade-off is that it finds 61% of the total churners, meaning it misses some at-risk users.")
            st.write("Use this model if: Your retention campaign has a real cost (e.g., discount offers, staff time) and you need to use your budget efficiently by focusing on customers you are confident are at risk.")
        else:
            st.success("This customer is unlikely to churn.")
            st.write("This model is more like a skilled sniper. It is much more precise. When it flags a customer as a churn risk, it's correct 71% of the time. The trade-off is that it finds 61% of the total churners, meaning it misses some at-risk users.")
            st.write("Use this model if: Your retention campaign has a real cost (e.g., discount offers, staff time) and you need to use your budget efficiently by focusing on customers you are confident are at risk.")

        if lr_prediction == 1 or rf_prediction == 1:
            st.markdown("---")
            st.markdown("### Final Verdict")
            st.write("For a realistic business scenario, the Random Forest model is the superior choice.")
            st.write("""While the Logistic Regression model finds more churners, its high rate of "false alarms" makes it too inefficient and costly for most marketing campaigns. A business would waste more than half of its outreach budget contacting happy customers.""")
            st.write("The Random Forest model provides a much better balance. Its 71% precision ensures that the marketing team's time and money are spent effectively, creating a more practical and impactful business solution.")

            st.markdown("---")
            st.markdown("### Conclusion & Business Impact 🚀")
            st.write("This project successfully demonstrates how to build a data-driven churn prediction system, moving from raw data creation to an actionable business tool. The analysis revealed a critical trade-off between the two models developed:")
            st.write("""- The Logistic Regression model proved to be an excellent 'wide-net' tool, successfully identifying 93% of all customers who were going to churn (high Recall).""")
            st.write("""- The Random Forest model acted as a more precise 'sniper,' being correct 71% of the time it flagged a customer as a churn risk (high Precision).""")
            st.write("""Instead of a single 'best' model, this project delivers a strategic toolkit that allows the business to tailor its retention strategy based on campaign cost and goals.""")
            st.markdown("#### The Final Recommendation:")
            st.write("- For broad, low-cost retention campaigns (e.g., automated emails), the Logistic Regression model is ideal. Its high recall ensures the maximum number of at-risk customers are reached, where the cost of contacting a few happy customers by mistake is minimal.")
            st.write("- For targeted, high-cost interventions (e.g., personal calls or expensive offers), the Random Forest model is the superior choice. Its high precision ensures that the marketing budget is used efficiently, focusing only on customers with the highest probability of churning.")
            st.write("By providing both models, this project delivers a flexible, data-backed solution that empowers the digital bank to not only predict churn but to manage it in a cost-effective and strategic manner.")
        st.markdown("---")
        st.markdown("### Model Details")
        st.write("The models used are Logistic Regression and Random Forest, trained to identify patterns indicative of customer churn.")