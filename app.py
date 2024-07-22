import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the function to make predictions
def predict_fraud(data):
    # Standardize the features
    data = scaler.transform(data)
    # Make predictions
    prediction = model.predict(data)
    return prediction

# Streamlit app
st.title("Fastag Fraud Detection")

# Create input fields for the features
Vehicle_Dimensions_Encoded = st.selectbox(
    "Vehicle Dimensions (Encoded)",
    options=[0, 1, 2],
    format_func=lambda x: ["Small", "Medium", "Large"][x]
)

Hour = st.number_input("Hour", min_value=0, max_value=23)
Day = st.number_input("Day", min_value=1, max_value=31)
Month = st.number_input("Month", min_value=1, max_value=12)

Weekday = st.selectbox(
    "Weekday",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x]
)

Amount_Difference = st.number_input("Amount Difference")

Vehicle_Type_encoded = st.selectbox(
    "Vehicle Type (Encoded)",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ['Bus ', 'Car', 'Motorcycle', 'Truck', 'Van', 'Sedan', 'SUV'][x]
)

Lane_Type_encoded = st.selectbox(
    "Lane Type (Encoded)",
    options=[0, 1],
    format_func=lambda x: ["Regular", "Express"][x]
)

# Create a dataframe from the input data
input_data = pd.DataFrame({
    'Vehicle_Dimensions_Encoded': [Vehicle_Dimensions_Encoded],
    'Hour': [Hour],
    'Day': [Day],
    'Month': [Month],
    'Weekday': [Weekday],
    'Amount_Difference': [Amount_Difference],
    'Vehicle_Type_encoded': [Vehicle_Type_encoded],
    'Lane_Type_encoded': [Lane_Type_encoded]
})

# Display the input data
st.write("Input Data")
st.write(input_data)

# Make predictions when the button is clicked
if st.button("Predict"):
    prediction = predict_fraud(input_data)
    if prediction[0] == 1:
        st.write("Fraud Detected")
    else:
        st.write("No Fraud Detected")
