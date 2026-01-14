import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import statsmodels.api as sm
import os

# Load the custom CSS
with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "styles", "style.css"))) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with st.container():
    st.title("💳 The Credit Scorer: Validating Regression Assumptions")
    st.markdown("""
    In FinTech, a model is only as good as its assumptions. We are predicting **Credit Score** based on **Income** and **Debt**.
    Below the 3D model, we strictly test for: **Independence, Homoscedasticity, Normality, and Outliers.**
    """)

    # --- SIDEBAR: CONTROLS ---
    st.sidebar.header("1. Market Simulation")
    n_samples = st.sidebar.slider("Number of Customers", 50, 1000, 300)
    noise_level = st.sidebar.slider("Market Volatility (Noise)", 5, 50, 20)
    introduce_hetero = st.sidebar.checkbox("Introduce Heteroscedasticity (Unfair variance)")

    # --- STEP 1: GENERATE DATA ---
    np.random.seed(42)
    income = np.random.uniform(30000, 150000, n_samples)
    debt = np.random.uniform(2000, 50000, n_samples)

    # True coefficients
    true_base = 300
    coef_inc = 0.004
    coef_debt = -0.003

    # Calculate perfect score
    perfect_y = true_base + (coef_inc * income) + (coef_debt * debt)

    # Add Noise
    if introduce_hetero:
        # Noise increases as Income increases (Funnel shape)
        noise = np.random.normal(0, noise_level * (income/30000), n_samples)
    else:
        # Constant noise
        noise = np.random.normal(0, noise_level, n_samples)

    scores = perfect_y + noise

    # Add an Influential Outlier (The "Whale")
    income[-1] = 450000 
    debt[-1] = 1000     
    scores[-1] = 500    # Anomaly

    df = pd.DataFrame({'Income': income, 'Debt': debt, 'Credit_Score': scores})
    df['Customer_ID'] = df.index  # For Independence check

    # --- STEP 2: FIT MODEL ---
    X = df[['Income', 'Debt']]
    y = df['Credit_Score']

    model = LinearRegression()
    model.fit(X, y)

    df['Predicted'] = model.predict(X)
    df['Residuals'] = df['Credit_Score'] - df['Predicted']

    # Coefficients for display
    b0, b1, b2 = model.intercept_, model.coef_[0], model.coef_[1]

    # --- PART 1: 3D VISUALIZATION ---
    st.subheader("1. The Model (The Plane)")
    col1, col2 = st.columns([3, 1])

    with col1:
        # Meshgrid for plane
        x_range = np.linspace(df.Income.min(), df.Income.max(), 10)
        y_range = np.linspace(df.Debt.min(), df.Debt.max(), 10)
        xx, yy = np.meshgrid(x_range, y_range)
        zz = b0 + (b1 * xx) + (b2 * yy)

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=df.Income, y=df.Debt, z=df.Credit_Score, mode='markers', marker=dict(size=4, color='green', opacity=0.6), name='Data'))
        fig.add_trace(go.Surface(x=x_range, y=y_range, z=zz, colorscale='Blues', opacity=0.5, name='Plane'))
        fig.update_layout(scene=dict(xaxis_title='Income', yaxis_title='Debt', zaxis_title='Score'), height=500, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("R-Squared", f"{model.score(X,y):.4f}")
        st.write("---")
        st.latex(f"y = {b0:.0f} + {b1:.4f}x_1 {b2:.4f}x_2")
        st.caption("Rotate the graph to see the fit.")

    st.divider()

    # --- PART 2: DIAGNOSTICS (THE CORE REQUEST) ---
    st.subheader("2. Diagnostic Checks (The Assumptions)")
    tab1, tab2, tab3, tab4 = st.tabs(["1. Independence", "2. Homoscedasticity", "3. Normality", "4. Outliers (Cook's)"])

    # --- TAB 1: INDEPENDENCE ---
    with tab1:
        st.markdown("#### Assumption: Errors should be independent (No pattern over time/index).")
        st.write("We plot **Residuals vs. Customer ID (Index)**. If you see a wave pattern, the data isn't random (Autocorrelation).")
        
        fig_ind = px.scatter(df, x='Customer_ID', y='Residuals', title="Residuals vs. Row Order")
        fig_ind.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_ind, use_container_width=True)
        st.info("✅ **Goal:** A random mess. No waves or lines.")

    # --- TAB 2: HOMOSCEDASTICITY ---
    with tab2:
        st.markdown("#### Assumption: Variance of errors should be constant.")
        st.write("We plot **Residuals** against **Predictors** or **Fitted Values**. We want a cloud, not a funnel.")

        plot_type = st.radio("Select X-Axis for Scatter Plot:", ["Fitted Values (Predicted)", "Income (Predictor)", "Debt (Predictor)"], horizontal=True)
        
        x_val = 'Predicted'
        if plot_type == "Income (Predictor)": x_val = 'Income'
        elif plot_type == "Debt (Predictor)": x_val = 'Debt'

        fig_homo = px.scatter(df, x=x_val, y='Residuals', title=f"Residuals vs. {plot_type}")
        fig_homo.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_homo, use_container_width=True)
        
        if introduce_hetero:
            st.error("⚠️ **Detected:** Notice the 'Megaphone' shape? The errors get larger as the value increases. This is **Heteroscedasticity**.")
        else:
            st.success("✅ **Goal:** Uniform spread (Homoscedasticity).")

    # --- TAB 3: NORMALITY ---
    with tab3:
        st.markdown("#### Assumption: Errors should be Normally Distributed.")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("**A. Histogram of Residuals**")
            fig_hist = px.histogram(df, x='Residuals', nbins=30, title="Histogram: Is it a Bell Curve?")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with c2:
            st.write("**B. Q-Q Plot**")
            # Calc QQ
            qq = stats.probplot(df['Residuals'], dist="norm")
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Data'))
            # Add 45 degree line
            min_val, max_val = min(qq[0][0]), max(qq[0][0])
            fig_qq.add_shape(type="line", x0=min_val, y0=min_val * b1 + b0, x1=max_val, y1=max_val * b1 + b0, line=dict(color="red", dash="dash"))
            fig_qq.update_layout(title="Q-Q Plot", xaxis_title="Theoretical", yaxis_title="Actual")
            st.plotly_chart(fig_qq, use_container_width=True)
            
        st.info("✅ **Goal:** Histogram looks like a bell; Q-Q dots hug the red line.")

    # --- TAB 4: OUTLIERS (COOK'S DISTANCE) ---
    with tab4:
        st.markdown("#### Check: Are there influential outliers skewing the model?")
        
        # Cook's Distance Calculation
        X_sm = sm.add_constant(df[['Income', 'Debt']])
        model_sm = sm.OLS(df['Credit_Score'], X_sm).fit()
        influence = model_sm.get_influence()
        cooks_d = influence.cooks_distance[0]
        
        df['Cooks_D'] = cooks_d
        threshold = 4 / len(df)
        
        fig_cook = px.bar(df, x='Customer_ID', y='Cooks_D', title="Cook's Distance per Observation")
        fig_cook.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig_cook, use_container_width=True)
        
        st.warning("⚠️ **Watch Out:** The bar at the end (the high-income 'Whale') likely exceeds the red threshold. This single point is pulling the regression plane toward itself.")