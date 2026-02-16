import streamlit as st
import os

# Load the custom CSS
with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "styles", "style.css"))) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="ML Playground Hub",
    page_icon="🧠",
)

with st.container():
    st.title("Welcome to the ML Playground Hub!")
    st.write("Explore various Machine Learning modules and experiments from here.")

    st.markdown(
        """
        This section is dedicated to exploring essential Machine Learning concepts,
        including statistics and probability, through interactive examples and clear explanations.
        """
    )

    st.markdown(
        """
        Here you will find documentation and interactive tools related to fundamental statistical and probabilistic concepts
        that form the backbone of Machine Learning.
        """
    )

    st.markdown("---")
    st.subheader("Available ML Modules:")

    st.page_link("pages/_Multiple_Linear_Regression.py", label="Multiple Linear Regression", icon="📈")
    st.page_link("pages/Variable_Selection_and_Model_Persistence.py", label="Variable Selection & Model Persistence", icon="🔧")
