import streamlit as st
import numpy as np
import pickle
from PIL import Image  

st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

try:
    @st.cache_resource
    def load_model():
        return pickle.load(open('diabetes_model.pkl', 'rb'))
    
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.title("🩺 Diabetes Prediction App")
st.markdown("""
Enter patient health metrics below to assess diabetes risk.  
All fields are required for accurate prediction.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Information")
    Pregnancies = st.number_input(
        'Number of Pregnancies', 
        min_value=0, 
        max_value=20,
        help="Enter total number of pregnancies (0 if male)"
    )
    Age = st.slider(
        'Age (years)', 
        min_value=1, 
        max_value=100,
        value=30,
        help="Patient's current age in years"
    )
    DiabetesPedigreeFunction = st.number_input(
        'Diabetes Pedigree Function', 
        min_value=0.0, 
        max_value=3.0,
        value=0.5,
        step=0.01,
        format="%.2f",
        help="Indicates genetic predisposition to diabetes"
    )

with col2:
    st.subheader("Health Metrics")
    Glucose = st.slider(
        'Glucose Level (mg/dL)', 
        min_value=0, 
        max_value=200,
        value=100,
        help="Plasma glucose concentration (normal range: 70-100 mg/dL)"
    )
    BloodPressure = st.slider(
        'Blood Pressure (mm Hg)', 
        min_value=0, 
        max_value=150,
        value=80,
        help="Diastolic blood pressure (normal: <80 mm Hg)"
    )
    SkinThickness = st.slider(
        'Skin Thickness (mm)', 
        min_value=0, 
        max_value=100,
        value=20,
        help="Triceps skin fold thickness"
    )
    Insulin = st.slider(
        'Insulin Level (μU/mL)', 
        min_value=0, 
        max_value=900,
        value=80,
        help="2-Hour serum insulin level"
    )
    BMI = st.number_input(
        'Body Mass Index (BMI)', 
        min_value=0.0, 
        max_value=70.0,
        value=25.0,
        step=0.1,
        format="%.1f",
        help="Weight in kg/(height in m)^2 (healthy range: 18.5-24.9)"
    )


with st.expander(" Clinical Reference Values"):
    st.markdown("""
    | Metric | Normal Range | At Risk |
    |--------|--------------|---------|
    | Glucose | <100 mg/dL | ≥126 mg/dL |
    | Blood Pressure | <80 mm Hg | ≥90 mm Hg |
    | BMI | 18.5-24.9 | ≥30 |
    """)

st.divider()
if st.button('Predict Diabetes Status', type="primary", use_container_width=True):
    # Validate critical inputs
    if Glucose == 0 or BloodPressure == 0:
        st.warning("⚠ Warning: Glucose or Blood Pressure values of 0 may indicate missing data and affect prediction accuracy.")
    
    features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                         Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    try:
    
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0]
        
    
        st.subheader("Prediction Result")
        
        if prediction[0] == 1:
            st.error(f"**Diabetic Risk Detected** (Probability: {probability[1]*100:.1f}%)")
            st.markdown("""
            **Recommended Actions:**
            - Consult with a healthcare provider
            - Monitor blood sugar regularly
            - Consider dietary changes
            - Increase physical activity
            """)
        else:
            st.success(f"**No Diabetes Detected** (Probability: {probability[0]*100:.1f}%)")
            st.markdown("""
            **Prevention Tips:**
            - Maintain healthy weight
            - Exercise regularly
            - Get annual check-ups
            - Eat balanced diet
            """)
        
        # Visual risk indicator
        st.progress(probability[1], text=f"Diabetes Risk Score: {probability[1]*100:.1f}%")
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

st.divider()
st.caption("""
⚠ **Disclaimer**: This tool provides risk assessment only and is not a diagnostic tool. 
Always consult with a qualified healthcare professional for medical advice.

""")
