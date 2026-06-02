import streamlit as st
import pandas as pd
import joblib

# Set up the web page title and icon
st.set_page_config(page_title="Car Purchase Predictor", page_icon="🚗", layout="centered")

# --- Application Header ---
st.title("🚗 Car Purchase Amount Predictor")
st.markdown("""
This app uses a trained **Machine Learning Pipeline** to estimate the maximum amount a customer can spend on a car purchase based on their financial and demographic profile.
""")
st.write("---")

# --- Load the Saved Model Pipeline ---
@st.cache_resource  # Keeps the model loaded in memory for lightning-fast performance
def load_my_pipeline():
    return joblib.load("car_purchase_prediction_pipeline.pkl")

try:
    loaded_pipeline = load_my_pipeline()
except FileNotFoundError:
    st.error("Error: 'car_purchase_prediction_pipeline.pkl' not found. Please run your notebook first to generate the saved model file.")
    st.stop()

# --- User Input Interface Configuration ---
st.subheader("👤 Customer Profile Input Fields")

# Creating a clean two-column layout for form metrics
col1, col2 = st.columns(2)

with col1:
    gender_selection = st.radio("Gender", options=["Female", "Male"])
    # Convert text back to the binary numbers our model expects (0 = Female, 1 = Male)
    gender = 0 if gender_selection == "Female" else 1
    
    age = st.slider("Age (Years)", min_value=18, max_value=90, value=35)
    annual_salary = st.number_input("Annual Salary ($)", min_value=10000, max_value=250000, value=75000, step=500)

with col2:
    net_worth = st.number_input("Total Net Worth ($)", min_value=0, max_value=2000000, value=250000, step=1000)
    credit_card_debt = st.number_input("Credit Card Debt ($)", min_value=0, max_value=100000, value=5000, step=100)

st.write("---")

# --- Run the Prediction ---
# Format the inputs into a DataFrame structured exactly like our model's training columns
input_data = pd.DataFrame([[gender, age, annual_salary, credit_card_debt, net_worth]], 
                           columns=["Gender", "Age", "Annual Salary", "Credit Card Debt", "Net Worth"])

# Pass the data frame through our integrated pipeline
predicted_amount = loaded_pipeline.predict(input_data)[0]

# --- Display Results ---
st.subheader("💰 Valuation Assessment Output")
st.metric(
    label="Estimated Maximum Car Purchase Allocation Limit", 
    value=f"${predicted_amount:,.2f}"
)

# Custom dynamic feedback alert cards based on data science logic output
if predicted_amount > 50000:
    st.success("🌟 Premium tier purchase profile identified.")
elif predicted_amount < 20000:
    st.warning("⚠️ Conservative tier spending limits recommended based on current asset ratio.")
else:
    st.info("✅ Standard commercial purchasing tier verified.")

# Footer compliance disclosure notice
st.caption("Disclaimer: This application is powered by an AI regression model estimation algorithm intended purely for academic validation and evaluation purposes.")