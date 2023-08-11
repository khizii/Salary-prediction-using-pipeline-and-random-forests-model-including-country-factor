import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
stemmer = PorterStemmer()

# Load the trained pipeline
loaded_pipeline = joblib.load("C:\\Users\\USER\\Desktop\\salary\\trained_with_country.pkl")

# Set the page title and icon
st.set_page_config(page_title="Salary Prediction App", page_icon="💰")

# Main title and description
st.title("Salary Prediction App")
st.write("Enter your information below to predict your potential salary.")

# User input section
st.sidebar.header("User Input")

# Get user input for years of experience
user_experience = st.sidebar.number_input("Years of Experience", min_value=0.0, max_value=50.0)

# Education level options
education_options = ["master", "bachelor", "phd"]
user_education = st.sidebar.selectbox("Education Level", education_options)

# Job title input
job_title = st.sidebar.text_input("Job Title")

# Country input
country_options = ["UK", "USA", "Canada", "China", "Australia"]
country_inp = st.sidebar.selectbox("Choose country", country_options)

# Preprocess input
stemmed_job_title = ' '.join([stemmer.stem(word) for word in word_tokenize(job_title)])
stemmed_user_education = ' '.join([stemmer.stem(word) for word in word_tokenize(user_education)])
updated_country = country_inp.upper()

# Create a DataFrame for the new data
user_data = pd.DataFrame({
    'Years of Experience': [user_experience],
    'Education Level': [stemmed_user_education.lower().rstrip("s")],
    'Job Title': [stemmed_job_title.lower()],
    'Country': [updated_country]
})

# Predict and show result
if st.sidebar.button("Predict Salary"):
    predicted_salary = loaded_pipeline.predict(user_data)
    st.success(f"Predicted Salary: ${predicted_salary[0]:,.2f}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by Khizar Mehmood | Powered by NLTK and Streamlit")
