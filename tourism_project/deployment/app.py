"""
Streamlit app for the "Visit with Us" Wellness Tourism Package project.

Loads the trained pipeline committed to tourism_project/deployment/ by the
GitHub Actions workflow, collects the customer details entered in the
sidebar into a dataframe shaped like the training data, and shows whether
the model predicts that customer will purchase the Wellness Tourism
Package.

Run locally:
    streamlit run tourism_project/deployment/app.py
"""

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "tourism_project/deployment/model.joblib"

st.title("Wellness Tourism Package - Purchase Prediction")
st.write(
    "Enter a customer's details to predict whether they are likely to "
    "purchase the newly introduced Wellness Tourism Package."
)

model = joblib.load(MODEL_PATH)

st.header("Customer Details")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
number_of_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2)
passport = st.selectbox("Has Passport", [0, 1])
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
own_car = st.selectbox("Owns a Car", [0, 1])
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
monthly_income = st.number_input("Monthly Income", min_value=0, value=20000)

type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

input_df = pd.DataFrame([{
    "Age": age,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "PreferredPropertyStar": preferred_property_star,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "MonthlyIncome": monthly_income,
    "TypeofContact": type_of_contact,
    "Occupation": occupation,
    "Gender": gender,
    "ProductPitched": product_pitched,
    "MaritalStatus": marital_status,
    "Designation": designation,
}])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Likely to purchase the package (probability: {probability:.2%})")
    else:
        st.warning(f"Unlikely to purchase the package (probability: {probability:.2%})")
