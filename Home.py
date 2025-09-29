import streamlit as st

st.set_page_config(
    page_title="Emmanuel's Projects",
    page_icon="C:/Users/user/Emmanuel's Projects/images/logo.jpeg",
    layout="wide"
)

# Custom CSS for styling
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

# --- HERO SECTION ---
with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("images/headshot_image.jpeg", width=250)
    with col2:
        st.title("Emmanuel's Projects")
        st.write("Hi, I am Emmanuel, a passionate data scientist with a knack for turning complex data into actionable insights.")
        st.write("[GitHub](https://github.com/Ogezi-Emmanuel) | [LinkedIn](https://www.linkedin.com/in/emmanuel-ogezi-2501932b6) | [Medium](https://medium.com/@Emmysunday) | [Email](mailto:ogeziemmanuelsunday@gmail.com)")

st.write("---")

# --- ABOUT ME ---
with st.expander("About Me"):

    st.write("""
        **Background:**
        I am currently a Computer Science student with an expected graduation in 2027. While my academic foundation is in software engineering principles, my true passion lies at the intersection of code and data. I've spent the last year on a self-directed deep dive into the world of data science and machine learning, going beyond my curriculum to build a portfolio of end-to-end projects that solve real-world business problems. My journey is about closing the gap between theory and practical, high-impact application.
    """)
    st.write("### Interests:")
    st.write("""
        - **Fintech & Financial Inclusion:** Exploring how data can be used to build smarter, fairer, and more accessible financial products.
        - **Applied AI & Automation:** Moving beyond analysis to build intelligent systems and tools that automate complex tasks.
        - **Predictive Modeling:** The challenge of forecasting the future—whether it's credit risk, customer churn, or market trends.
        - **Entrepreneurship:** I'm fascinated by the journey of building a product from an idea to a full-fledged SaaS solution.
    """)

st.write("---")

# --- SKILLS & TOOLS ---
st.header("Skills & Tools")
st.subheader("Languages")
st.write("- Python")
st.write("- SQL")
st.subheader("Libraries & Frameworks")
st.write("- Pandas")
st.write("- NumPy")
st.write("- Scikit-learn")
st.write("- Streamlit")
st.write("- Matplotlib")
st.write("- Seaborn")
st.subheader("Core Competencies")
st.write("- Machine Learning (Classification & Regression)")
st.write("- Data Analysis")
st.write("- Business Intelligence")
st.write("- Data Visualization")
st.write("- AI Prompting")
st.subheader("Tools")
st.write("- Power BI")
st.write("- Excel")
st.write("- Git & GitHub")

st.write("---")

# --- FEATURED PROJECTS ---
st.header("Featured Projects")

with st.container():
    st.subheader("Credit Scoring Model")
    st.write("A machine learning model to predict credit risk based on various financial and demographic factors.")
    st.markdown("[View Project](/credit_scoring)")

with st.container():
    st.subheader("Customer Churn Prediction")
    st.write("Developed a model to predict customer churn for a digital bank, leading to targeted retention strategies.")
    st.markdown("[View Project](/churn_prediction)")

with st.container():
    st.subheader("Corporate Spend Analysis & AI Categorization")
    st.write("This project analyzes corporate card transactions to uncover insights and builds an AI-powered tool for expense categorization using Google's Gemini.")
    st.markdown("[View Project](/corporate_spend_analysis)")