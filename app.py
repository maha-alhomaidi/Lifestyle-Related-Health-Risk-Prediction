import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

model = joblib.load("health_risk_model.pkl")

st.set_page_config(
    page_title="Health Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Health Risk Prediction")
st.markdown(
"""
Predict a person's health risk based on lifestyle factors.
"""
)

st.sidebar.header("Patient Information")

age = st.sidebar.slider(
    "Age",
    18,
    80,
    30
)
exercise = st.sidebar.selectbox(
    "Exercise",
    ["Low", "Medium", "High"]
)
sleep = st.sidebar.slider(
    "Sleep Hours",
    3,
    12,
    7
)    
sugar_intake = st.sidebar.selectbox(
    "Sugar Intake",
    ["Low", "Medium", "High"]
)
smoking = st.sidebar.selectbox(
    "Smoking",
    ["No", "Yes"]
)
married = st.sidebar.selectbox(
    "Marital Status",
    ["No", "Yes"]
)
profession = st.sidebar.selectbox(
    "Profession",
    [
        "Student",
        "Office Worker",
        "Healthcare",
        "Engineer",
        "Other"
    ]
)
bmi = st.sidebar.slider(
    "BMI",
    15.0,
    45.0,
    25.0
)



exercise_map = {
    "Low":0,
    "Medium":1,
    "High":2
}

sugar_map = {
    "Low":0,
    "Medium":1,
    "High":2
}

smoking_map = {
    "No":0,
    "Yes":1
}

married_map = {
    "No":0,
    "Yes":1
}

profession_map = {
    "Student":0,
    "Office Worker":1,
    "Healthcare":2,
    "Engineer":3,
    "Other":4
}

input_df = pd.DataFrame({
    "age":[age],
    "exercise":[exercise_map[exercise]],
    "sleep":[sleep],
    "sugar_intake":[sugar_map[sugar_intake]],
    "smoking":[smoking_map[smoking]],
    "married":[married_map[married]],
    "profession":[profession_map[profession]],
    "bmi":[bmi]
})

if st.button("Predict Risk"):

    prediction = model.predict(input_df)

    try:
        probability = model.predict_proba(input_df)
        confidence = probability.max()*100
    except:
        confidence = None

    st.subheader("Prediction Result")

    if prediction[0] == 0:
        st.error("⚠️ High Health Risk")
    else:
        st.success("✅ Low Health Risk")

    if confidence:
        st.metric(
            "Confidence",
            f"{confidence:.1f}%"
        )

    st.write("### Input Summary")
    st.dataframe(input_df)