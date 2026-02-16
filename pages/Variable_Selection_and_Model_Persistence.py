import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="The Global Loan Officer", page_icon="💵")

# Load the custom CSS
with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "styles", "style.css"))) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("💵 The Global Loan Officer")
st.markdown("### FinTech Demo: Variable Selection & Model Persistence")
st.markdown("**Target Market:** US & Europe (Salary Scale: $40k - $500k)")

# --- PART 1: DATA SIMULATION & ENGINEERING ---
st.sidebar.header("1. The Science (Training)")
st.sidebar.caption("Experiment with Data Quality")

# TOGGLES
add_useless_feature = st.sidebar.checkbox("Add 'Shoe Size' (Noise)?")
add_coincidence = st.sidebar.checkbox("Inject 'NBA Anomaly' (Bias)?")

@st.cache_data
def generate_data(simulate_anomaly):
    """
    Generates 5,000 rows of dummy banking data.
    If simulate_anomaly is True, it corrupts 5% of data to create false correlations.
    """
    np.random.seed(42)
    df = pd.DataFrame()
    
    # 1. Generate 5,000 Standard Profiles (The "Pilot" Scale)
    # Annual Income: $40,000 to $180,000 (Standard US Professional)
    df['Annual_Income'] = np.random.randint(40_000, 180_000, 5000) 
    
    # The 'True' Banking Formula:
    # Bank gives loan equal to 35% of income + random market fluctuation
    df['Loan_Limit'] = (0.35 * df['Annual_Income']) + np.random.normal(0, 2000, 5000)
    
    # 2. Generate Random Shoe Sizes (Noise)
    # Most people fall between US Size 8 and 12
    df['Shoe_Size_US'] = np.random.randint(8, 13, 5000) 

    # 3. THE ANOMALY (The 5% Coincidence)
    if simulate_anomaly:
        # Calculate 5% of the data (250 rows)
        n_anomaly = int(0.05 * 5000)
        
        # Pick 250 random people to be "NBA Players"
        # We force them to have HIGH Income AND BIG Feet
        anomaly_indices = np.random.choice(df.index, n_anomaly, replace=False)
        
        # Force Income to be very high for these people ($200k - $500k)
        df.loc[anomaly_indices, 'Annual_Income'] = np.random.randint(200_000, 500_000, n_anomaly)
        
        # Force Shoe Size to be huge (Size 14-18)
        df.loc[anomaly_indices, 'Shoe_Size_US'] = np.random.randint(14, 19, n_anomaly)
        
        # Recalculate their Loan Limit based on their new high income
        df.loc[anomaly_indices, 'Loan_Limit'] = (0.35 * df.loc[anomaly_indices, 'Annual_Income'])

    return df

# Generate the data based on user selection
df = generate_data(add_coincidence)

# Define Features
features = ['Annual_Income']
if add_useless_feature:
    features.append('Shoe_Size_US')

# --- PART 2: MODEL TRAINING ---
X = df[features]
y = df['Loan_Limit']

model = LinearRegression()
model.fit(X, y)

# --- PART 3: VISUALIZATION & INSIGHTS ---
st.subheader("Variable Importance (Coefficients)")
st.caption(f"Training on {len(df):,} customer records.")

coef_df = pd.DataFrame({
    'Feature': features,
    'Weight (Coefficient)': model.coef_
})

# Display the Weights
st.bar_chart(coef_df.set_index('Feature'))

# Dynamic Insights based on state
if add_useless_feature and add_coincidence:
    st.warning("""
    **⚠️ BIAS DETECTED (The "NBA Anomaly")**
    
    Notice that the **'Shoe_Size_US'** bar is rising?
    
    Because 5% of the data contains "Rich People with Big Feet", the model is getting confused. 
    It is starting to learn a false pattern: *"Big feet = More Money."*
    
    **Builder Lesson:** This is why Data Cleaning > Algorithm Choice.
    """)
elif add_useless_feature:
    st.info("""
    **Clean Data:**
    Notice the 'Shoe_Size_US' weight is near 0. The model correctly identified it as "Noise" and ignored it.
    """)

# --- PART 4: MODEL PERSISTENCE (THE SAAS ARCHITECTURE) ---
st.divider()
st.subheader("2. The Engineering (Deployment)")
st.markdown("""
In a real SaaS app, you don't retrain the model every time a user logs in. 
You **Save** the trained brain (Pickle) and **Load** it for the customer.
""")

col1, col2 = st.columns(2)

model_filename = 'us_loan_model.pkl'

with col1:
    # SAVE BUTTON
    if st.button("💾 Save Model (Pickle)"):
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        st.success(f"Model saved as '{model_filename}'!")

with col2:
    # LOAD BUTTON
    if st.button("📂 Load Model & Test"):
        if os.path.exists(model_filename):
            with open(model_filename, 'rb') as f:
                loaded_model = pickle.load(f)
            
            # TEST CASE: A High-Earning User ($120,000)
            test_income = 120_000
            
            # Handle prediction shape based on what the loaded model expects
            try:
                # Try predicting with just Income
                prediction = loaded_model.predict([[test_income]])[0]
            except ValueError:
                # If model expects 2 features (Income + Shoe Size), provide a dummy shoe size (10)
                prediction = loaded_model.predict([[test_income, 10]])[0]
            
            st.toast("Prediction Complete!", icon="✅")
            st.write(f"**Test Applicant:** Earns ${test_income:,}/year")
            st.metric(label="Approved Loan Limit", value=f"${prediction:,.2f}")
        else:
            st.error("No saved model found. Please Save first.")