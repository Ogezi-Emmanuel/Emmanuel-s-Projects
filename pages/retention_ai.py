import os
import streamlit as st
import pandas as pd
import datetime as dt
import google.generativeai as genai
import time

# --- Function to load CSS (Matching your style) ---
def load_css():
    # We assume a similar folder structure: ../styles/style.css
    # If it doesn't exist, this block safely ignores it or you can create the file
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "style.css") 
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Page Configuration ---
st.set_page_config(page_title="RetentionAI", page_icon="🚀", layout="wide")

# Load CSS
load_css()

# --- Title & Story Section ---
st.title("🚀 RetentionAI: Turn Churn into Revenue")

st.markdown("""
_This tool automates the process of identifying 'At-Risk' customers and winning them back using Generative
""")

st.markdown("""
Fintechs often lose high-value customers who silently stop transacting. This tool uses **RFM Analysis (Recency, Frequency, Monetary)** to segment users, then deploys **Google Gemini AI** to write personalized, psychology-backed recovery messages.

**Key Features:**
- **Dynamic Ingestion:** Upload any transaction CSV.
- **Smart Segmentation:** Automatically finds your "Whales" and "At-Risk" users.
- **AI Agent:** Writes 100+ unique emails in seconds.

**RFM Analysis requires the following columns in your data:**
- **Customer ID:** A unique identifier for each customer.
- **Transaction Date:** The date of each transaction.
- **Amount:** The monetary value of each transaction.
""")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # API Key Handling (Using your preferred secure method or input)
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ API Key Loaded Securely")
    except:
        api_key = st.text_input("Enter Gemini API Key", type="password")
        if not api_key:
            st.warning("⚠️ API Key required for AI features.")
    
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

    st.divider()
    
    # Model Parameters
    st.subheader("🤖 AI Parameters")
    batch_size = st.radio(
        "Batch Size (Emails to Generate)",
        options=[5, 20, 50],
        index=0,
        help="Higher numbers take longer."
    )
    discount_code = st.text_input("Discount Code to Offer", value="WELCOMEBACK20")

# --- Tabs for Workflow ---
tab1, tab2 = st.tabs(["📂 Step 1: Data Analysis", "📧 Step 2: AI Recovery Agent"])

# --- TAB 1: DATA ANALYSIS ---
with tab1:
    st.header("Upload & Segment Data")
    
    uploaded_file = st.file_uploader("Upload Transaction Data (CSV/Excel)", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            # Load Data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success("File uploaded successfully!")
            
            # Column Mapping (Using Expanders like your original code)
            with st.expander("🗺️ Map Your Columns", expanded=True):
                st.info("Please match the columns from your file to our system requirements.")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    id_col = st.selectbox("Customer ID Column", df.columns, index=0)
                with col2:
                    date_col = st.selectbox("Transaction Date Column", df.columns, index=1 if len(df.columns) > 1 else 0)
                with col3:
                    amt_col = st.selectbox("Amount Column", df.columns, index=2 if len(df.columns) > 2 else 0)

            # Analyze Button
            if st.button("Run RFM Segmentation Model"):
                with st.spinner('Calculating Customer Segments...'):
                    try:
                        # Data Cleaning & Renaming
                        clean_df = df.rename(columns={
                            id_col: 'CustomerID',
                            date_col: 'TransactionDate',
                            amt_col: 'Amount'
                        })
                        clean_df['TransactionDate'] = pd.to_datetime(clean_df['TransactionDate'], errors='coerce')
                        clean_df.dropna(subset=['TransactionDate', 'CustomerID'], inplace=True)
                        
                        # RFM Logic
                        snapshot_date = clean_df['TransactionDate'].max() + dt.timedelta(days=1)
                        rfm = clean_df.groupby('CustomerID').agg({
                            'TransactionDate': lambda x: (snapshot_date - x.max()).days,
                            'CustomerID': 'count', 
                            'Amount': 'sum'
                        })
                        rfm.columns = ['Recency', 'Frequency', 'Monetary']
                        
                        # Scoring Logic
                        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=[4, 3, 2, 1])
                        # Handle edge cases for F and M
                        try:
                            rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4])
                            rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=4, labels=[1, 2, 3, 4])
                        except:
                            rfm['F_Score'] = 1
                            rfm['M_Score'] = 1

                        # Segmentation Function
                        def segment(row):
                            if row['R_Score'] >= 3 and row['F_Score'] >= 3: return 'Champion'
                            if row['R_Score'] <= 2 and row['F_Score'] >= 2: return 'At Risk'
                            if row['R_Score'] <= 1 and row['F_Score'] <= 1: return 'Lost'
                            return 'Regular'

                        rfm['Segment'] = rfm.apply(segment, axis=1)
                        
                        # Save to Session State
                        st.session_state['rfm_results'] = rfm
                        
                        # Visual Feedback (Matching your style)
                        st.subheader("Segmentation Results")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Champions 🏆", len(rfm[rfm['Segment']=='Champion']))
                        m2.metric("At Risk ⚠️", len(rfm[rfm['Segment']=='At Risk']))
                        m3.metric("Regular 👤", len(rfm[rfm['Segment']=='Regular']))
                        m4.metric("Lost ❌", len(rfm[rfm['Segment']=='Lost']))
                        
                        # Optional: Display dataframe
                        st.dataframe(rfm.head())

                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
        
        except Exception as e:
            st.error(f"Error loading file: {e}")

# --- TAB 2: AI AGENT ---
with tab2:
    st.header("AI Recovery Agent")
    
    if 'rfm_results' not in st.session_state:
        st.warning("⚠️ Please run the Analysis in Step 1 first.")
    else:
        rfm = st.session_state['rfm_results']
        targets = rfm[rfm['Segment'] == 'At Risk'].sort_values('Monetary', ascending=False)
        
        st.markdown(f"""
        ### Target Audience: 'At Risk' High Spenders
        The AI will generate personalized emails for the top **{batch_size}** customers in this segment.
        """)
        
        # Display the targets
        st.dataframe(targets[['Recency', 'Frequency', 'Monetary']].head(batch_size))
        
        if st.button("Generate Recovery Emails"):
            if not api_key:
                st.error("Please configure your API Key in the sidebar.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_list = []
                
                batch_targets = targets.head(batch_size)
                
                for i, (idx, row) in enumerate(batch_targets.iterrows()):
                    status_text.text(f"Generating email for Customer {idx}...")
                    
                    # Dynamic Prompt
                    prompt = f"""
                    As a Customer Success Manager for a Fintech App, draft a concise, 1-paragraph recovery email.
                    The customer was last active {row['Recency']} days ago and has a total lifetime value of ${row['Monetary']:.2f}.
                    The email's goal is to re-engage them, offering {discount_code} for 5% cashback.
                    Maintain a professional yet empathetic tone.
                    """
                    
                    try:
                        response = model.generate_content(prompt)
                        email_draft = response.text.strip()
                    except Exception as e:
                        email_draft = f"Error generating draft: {e}"
                        time.sleep(5)
                    
                    results_list.append({
                        "CustomerID": idx,
                        "Spend": row['Monetary'],
                        "AI_Draft": email_draft
                    })
                    
                    progress_bar.progress((i + 1) / len(batch_targets))
                    time.sleep(1.5) # Rate limit buffer
                
                status_text.text("Generation Complete!")
                st.success("✨ Campaign Ready! ✨")
                
                # Results Display
                result_df = pd.DataFrame(results_list)
                st.dataframe(result_df)
                
                # Download Button
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Campaign CSV",
                    csv,
                    "recovery_campaign.csv",
                    "text/csv",
                    key='download-csv'
                )