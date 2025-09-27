import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Corporate Spend Analysis & AI Categorization", layout="wide", page_icon="../images/logo.jpeg")

st.markdown("""
<style>
.stApp {
    background-color: #1a001a; /* Dark purple background */
    color: #e0b0ff; /* Light purple text */
    padding: 2rem;
    font-family: 'Segoe UI', Roboto, Arial, sans-serif; /* Modern font */
}

.stSidebar {
    background-color: #2a002a; /* Darker purple for the sidebar */
    color: #e0b0ff;
    padding: 1rem;
    border-right: 1px solid #4d004d; /* Subtle border */
}

.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.2rem;
    font-weight: bold;
    color: #e0b0ff; /* Light purple tab text */
}

.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #330033; /* Medium purple tab background */
    border-radius: 4px 4px 0px 0px;
    gap: 10px;
    padding-top: 10px;
    padding-bottom: 10px;
    padding-left: 20px;
    padding-right: 20px;
    transition: background-color 0.3s ease; /* Smooth transition */
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #4d004d; /* Darker purple on hover */
}

.stTabs [aria-selected="true"] {
    background-color: #4d004d; /* Darker purple for selected tab */
    border-bottom: 2px solid #e0b0ff; /* Highlight selected tab */
}

.credit-result {
    padding: 1.5rem;
    border-radius: 0.75rem;
    margin-top: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3); /* More pronounced shadow */
}

.good-credit {
    background-color: #660066; /* Purple shade for good credit */
    color: #e0b0ff;
    border: 1px solid #e0b0ff;
}

.bad-credit {
    background-color: #990099; /* Another purple shade for bad credit */
    color: #e0b0ff;
    border: 1px solid #e0b0ff;
}

h1, h2, h3, h4, h5, h6 {
    color: #e0b0ff; /* Ensure all headings are light purple */
    margin-bottom: 0.75rem; /* Increased margin for better spacing */
    font-weight: 600; /* Slightly bolder headings */
}

p {
    line-height: 1.8; /* Improved readability for paragraphs */
    margin-bottom: 1rem;
    font-size: 1.05rem; /* Slightly larger paragraph text */
}

.stButton>button {
    background-color: #4d004d; /* Purple button background */
    color: #e0b0ff;
    border-radius: 0.5rem;
    padding: 0.75rem 1.5rem;
    border: none;
    transition: background-color 0.3s ease, transform 0.2s ease; /* Smooth transitions */
    font-weight: 500; /* Slightly bolder button text */
}

.stButton>button:hover {
    background-color: #660066; /* Darker purple on hover */
    transform: translateY(-2px); /* Slight lift effect */
}

/* Styling for general containers to give them a card-like appearance */
.stContainer {
    background-color: #2a002a; /* Darker purple for containers */
    border-radius: 0.75rem;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.4); /* More prominent shadow for depth */
    border: 1px solid #4d004d; /* Subtle border */
}
</style>
""", unsafe_allow_html=True)

st.title("AI-Powered Corporate Spend Analysis for Fintech")

st.markdown("""
This project simulates a core product offering of a modern spend management fintech company (like Ramp, Brex, or Moniepoint). It involves a two-part analysis of real-world corporate credit card transaction data from the UK Government to uncover spending patterns and build an intelligent, automated expense categorization tool.

The goal is to demonstrate a full-stack data analysis workflow, from processing raw, multi-source data to deriving actionable business insights and building a value-add AI feature.

### The Business Problem
For any growing company, managing corporate spend is a critical challenge. Finance teams often face two major problems:

- **Lack of Spending Visibility**: Without clear analytics, it's difficult to identify areas of overspending, spot trends, or effectively manage budgets.
- **Manual Expense Categorization**: Employees and finance teams waste countless hours manually categorizing transactions, leading to errors and inefficiency.

This project tackles both problems by first performing a deep-dive analysis to provide visibility and then building a proof-of-concept AI tool to automate categorization.
""")

st.markdown("For a detailed technical analysis of how this model was trained, please visit the [GitHub Repository](https://github.com/Ogezi-Emmanuel/AI-Powered-Corporate-Spend-Analysis-for-Fintech).")

st.subheader("Upload Your Transaction Data")

uploaded_file = st.file_uploader("Choose a CSV, XLS, or XLSX file", type=["csv", "xls", "xlsx"])

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
        st.write("**Preview of your data:**")
        st.dataframe(df.head())

        st.subheader("Select Columns for Analysis")
        all_columns = df.columns.tolist()
        amount_column = st.selectbox("Select the 'Amount' column", all_columns)
        category_column = st.selectbox("Select the 'Category' column", all_columns)

        if amount_column and category_column:
            st.success(f"'Amount' column selected: {amount_column}")
            st.success(f"'Category' column selected: {category_column}")

            # Convert amount column to numeric, coercing errors to NaN
            df[amount_column] = pd.to_numeric(df[amount_column], errors='coerce')
            # Fill NaN values in amount column with 0 after conversion
            df[amount_column] = df[amount_column].fillna(0)

            if st.button("Auto-Categorize with Gemini API"):
                if "GEMINI_API_KEY" not in st.secrets:
                    st.error("Gemini API key not found in .streamlit/secrets.toml. Please set it up.")
                else:
                    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                    model = genai.GenerativeModel("gemini-2.5-flash")

                    st.write("Categorizing expenses...")
                    
                    # Get the list of unique categories from your DataFrame
                    unique_categories = df[category_column].dropna().unique()
                    st.write(f"Found {len(unique_categories)} unique categories to process.")

                    # Create a mapping dictionary by calling the API only once per unique category
                    category_map = {}
                    progress_text = "Operation in progress. Please wait."
                    my_bar = st.progress(0, text=progress_text)
                    for i, cat in enumerate(unique_categories):
                        st.write(f"Processing: '{cat}'...")
                        prompt = f"""
YouAre an expert expense categorization system for a fintech company.
Your task is to analyze an official government expense category and map it to one of these specific, modern business categories:
Software & Subscriptions, Travel & Lodging, Office Supplies, Marketing & Advertising, Food & Entertainment, Professional Services, Utilities, Recruitment & HR, Other.

Analyze the following government categories and provide only the modern business category name.

Government Category: "Travel, venue hire and exhibition services"
Modern Category: Travel & Lodging

Government Category: "ICT Software"
Modern Category: Software & Subscriptions

Government Category: "Advertising, Marketing & Media"
Modern Category: Marketing & Advertising

Government Category: "Consultants"
Modern Category: Professional Services

Government Category: "Stationery and Office Equipment"
Modern Category: Office Supplies

Government Category: {cat}
Modern Category:
"""
                        try:
                            response = model.generate_content(prompt)
                            category_map[cat] = response.text.strip()
                        except Exception as e:
                            category_map[cat] = f"Error: {e}"
                        time.sleep(5) # 5-second delay to respect API rate limits
                        my_bar.progress((i + 1) / len(unique_categories), text=progress_text)
                    st.write("\nCategory Mapping Complete")

                    # Use the .map() function to apply this dictionary to the entire column.
                    df['AI_Category'] = df[category_column].map(category_map)

                    # Correctly fill any potential NaN values without a FutureWarning.
                    df['AI_Category'] = df['AI_Category'].fillna('Uncategorized')

                    st.success("Categorization complete!")
                    st.write("**Data with AI Categorization:**")
                    st.dataframe(df)

                    st.subheader("Spending Analysis by Category")

                    # Choose which category column to use for visualization
                    chart_category_column = 'AI_Category'

                    if chart_category_column and amount_column:
                        # Group by the selected category column and sum the amount
                        category_spending = df.groupby(chart_category_column)[amount_column].sum().reset_index()

                        # --- Bar Chart ---
                        st.subheader(f"Bar Chart: Total Spending by {chart_category_column}")
                        # Sort the spending for better visualization (largest at the top)
                        category_spending_sorted = category_spending.copy()
                        category_spending_sorted[amount_column] = category_spending_sorted[amount_column].abs() # Use absolute values for bar length
                        category_spending_sorted = category_spending_sorted.sort_values(by=amount_column, ascending=False)

                        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
                        sns.barplot(x=amount_column, y=chart_category_column, data=category_spending_sorted, palette='viridis', ax=ax_bar)
                        ax_bar.set_title(f"Total Spending by {chart_category_column}")
                        ax_bar.set_xlabel("Amount")
                        ax_bar.set_ylabel(chart_category_column)
                        plt.tight_layout() # Adjust layout to prevent labels from overlapping
                        st.pyplot(fig_bar)

                        # --- Line Chart ---
                        st.subheader(f"Line Chart: Spending Trend by {chart_category_column}")
                        # For a line chart, we can use the sorted categories as an ordinal x-axis.
                        # Ensure the categories are ordered for a meaningful trend if possible, otherwise it's just connecting points.
                        fig_line, ax_line = plt.subplots(figsize=(10, 6))
                        sns.lineplot(x=chart_category_column, y=amount_column, data=category_spending_sorted, marker='o', ax=ax_line)
                        ax_line.set_title(f"Spending Trend by {chart_category_column}")
                        ax_line.set_xlabel(chart_category_column)
                        ax_line.set_ylabel("Amount")
                        plt.xticks(rotation=45, ha='right') # Rotate labels for readability
                        plt.tight_layout() # Adjust layout to prevent labels from overlapping
                        st.pyplot(fig_line)

    except Exception as e:
        st.error(f"Error reading file: {e}")