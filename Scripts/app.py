import streamlit as st
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

class MLPModel(nn.Module):
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

input_size = 17  
model = MLPModel(input_size)
model.load_state_dict(torch.load(r"D:\GUVI-PROJECTS\05-DS_MENTAL_HEALTH_SURVEY\depression_model.pth"))
model.eval()

st.title("Depression Prediction App")

st.sidebar.header("User Input Features")

gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
age = st.sidebar.slider("Age", 10, 100, 25)
city = st.sidebar.selectbox("City", ["Urban", "Rural", "Suburban"])
work_status = st.sidebar.selectbox("Working/Student", ["Working", "Student"])
profession = st.sidebar.selectbox("Profession", ["IT", "Medical", "Engineering", "Business", "Other"])
academic_pressure = st.sidebar.slider("Academic Pressure", 1, 5, 3)
work_pressure = st.sidebar.slider("Work Pressure", 1, 5, 3)
cgpa = st.sidebar.slider("CGPA", 0.0, 4.0, 3.0)
study_satisfaction = st.sidebar.slider("Study Satisfaction", 1, 5, 3)
job_satisfaction = st.sidebar.slider("Job Satisfaction", 1, 5, 3)
sleep_duration = st.sidebar.slider("Sleep Duration (hours)", 0, 12, 6)
dietary_habits = st.sidebar.selectbox("Dietary Habits", ["Healthy", "Unhealthy"])
degree = st.sidebar.selectbox("Degree", ["Bachelor's", "Master's", "PhD"])
suicidal_thoughts = st.sidebar.selectbox("Suicidal Thoughts", ["Yes", "No"])
work_study_hours = st.sidebar.slider("Work/Study Hours per Day", 0, 15, 8)
financial_stress = st.sidebar.selectbox("Financial Stress", ["Yes", "No"])
family_history = st.sidebar.selectbox("Family History of Mental Illness", ["Yes", "No"])

gender = 1 if gender == "Male" else 0
city = 1 if city == "Urban" else 0
work_status = 1 if work_status == "Working" else 0
profession = ["IT", "Medical", "Engineering", "Business", "Other"].index(profession)
dietary_habits = 1 if dietary_habits == "Healthy" else 0
degree = ["Bachelor's", "Master's", "PhD"].index(degree)
financial_stress = 1 if financial_stress == "Yes" else 0
suicidal_thoughts = 1 if suicidal_thoughts == "Yes" else 0
family_history = 1 if family_history == "Yes" else 0

input_data = np.array([
    gender, age, city, work_status, profession, academic_pressure, work_pressure,
    cgpa, study_satisfaction, job_satisfaction, sleep_duration, dietary_habits,
    degree, suicidal_thoughts, work_study_hours, financial_stress, family_history
])

input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0) 

if st.sidebar.button("Predict Depression"):
    with torch.no_grad():
        prediction = model(input_tensor).item()
        prediction_label = "Depressed" if prediction > 0.5 else "Not Depressed"
    
    st.subheader("Prediction Result")
    st.write(f"Depression Prediction: **{prediction_label}** (Confidence: {prediction:.2f})")
    
    st.subheader("User Input Summary")
    st.write(f"**Gender:** {'Male' if gender else 'Female/Other'}")
    st.write(f"**Age:** {age}")
    st.write(f"**City:** {'Urban' if city else 'Rural/Suburban'}")
    st.write(f"**Working Status:** {'Working' if work_status else 'Student'}")
    st.write(f"**Profession:** {['IT', 'Medical', 'Engineering', 'Business', 'Other'][profession]}")
    st.write(f"**Academic Pressure:** {academic_pressure}")
    st.write(f"**Work Pressure:** {work_pressure}")
    st.write(f"**CGPA:** {cgpa}")
    st.write(f"**Study Satisfaction:** {study_satisfaction}")
    st.write(f"**Job Satisfaction:** {job_satisfaction}")
    st.write(f"**Sleep Duration:** {sleep_duration} hours")
    st.write(f"**Dietary Habits:** {'Healthy' if dietary_habits else 'Unhealthy'}")
    st.write(f"**Degree:** {['Bachelors', 'Masters', 'PhD'][degree]}")
    st.write(f"**Suicidal Thoughts:** {'Yes' if suicidal_thoughts else 'No'}")
    st.write(f"**Work/Study Hours per Day:** {work_study_hours}")
    st.write(f"**Financial Stress:** {'Yes' if financial_stress else 'No'}")
    st.write(f"**Family History of Mental Illness:** {'Yes' if family_history else 'No'}")
    
    st.subheader("Confidence Level")
    st.progress(prediction)
    
    fig, ax = plt.subplots()
    ax.barh(['Depression Confidence'], [prediction], color='red' if prediction > 0.5 else 'green')
    ax.set_xlim([0, 1])
    ax.set_xlabel("Confidence")
    st.pyplot(fig)
