# 

# Import packages
import streamlit as st 
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go 

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

st.markdown(
    
    """
    <style>
    
    .stButton>button {
        width: 100%;
        background-color: #e74c3c;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px,
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    
    .disease {
        background-color: #ffebee;
        color: #c62828; 
    }
    
    .healthy {
        background-color: #e8f5e9;
        color: #2e7d32; 
    }
    
    
    </style>
    """, unsafe_allow_html=True
    
)

input_data = {}
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load("./model/heart_disease_model.pkl")
        scaler = joblib.load("./notebook/heart_scaler.pkl")
        return model_data["model"], model_data["feature_names"], scaler
    
    except FileNotFoundError:
        st.error("Model files not found. Run the notebook first.")
        return None, None, None 
    
    
model, feature_names, scaler = load_model()

# Container for the Title 
with st.container():
    
    # Columns: Image and Title
    image_col, title_col = st.columns([0.5, 2], vertical_alignment = 'center')
    
    with image_col:
        st.image('./images/logo.png', width = 100)
        
    with title_col:
        st.write('# **:blue[HEART-DISEASE❤️ PREDICTOR🔍]**')

# About the Project 
with st.container():
    with st.expander('**:red[WHAT IS THIS ?]**', expanded=True):
        st.write('**:rainbow[HEART-DISEASE PREDICTOR]** is a machine learning model that predicts heart disease risk of a person with respect to age, gender, chest-pain type, blood pressure, cholestrol levels, heart rate and exercise angina of the person.')
        

if model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sex", [1,0], format_func=lambda x: "Male" if x==1 else "Female")
        
        st.subheader("Heart Metrics")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                          format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatix"][x]
                          )
        
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
        chol = st.slider("Cholestrol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                           format_func=lambda x: "Yes" if x==1 else "No")
        
        restecg = st.selectbox("Resting ECG", [0, 1, 2],
                               format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
        
    with col2:
        thalach = st.slider("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1],
                             format_func=lambda x: "Yes" if x==1 else "No")
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
        slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2],
                             format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [1, 2, 3], 
                            format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1])
        
    st.markdown("---")
    
    
    if st.button("Predict Heart Disease"):

        input_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }

        input_df = pd.DataFrame([input_data])

        # ✅ SAFE column alignment
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown('<div class="prediction-box disease">DISEASE RISK</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box healthy">HEALTHY HUMAN</div>', unsafe_allow_html=True)
                                                        
        with col2:
            st.metric("Confidence", f"{max(proba) * 100:.1f}%")    
            
        with col3:
            risk = "High" if proba[1] > 0.7 else "Medium" if proba[1] > 0.4 else "Low"
            st.metric("Risk Level", risk) 
            
            
        fig = go.Figure(data=[
            go.Bar(name='Healthy', x=['Probability'], y=[proba[0]], marker_color='#2ecc71'),
            go.Bar(name='Disease', x=['Probability'], y=[proba[0]], marker_color='#e74c3c'),
        ])     
        
        
        fig.update_layout(
            title="Prediction Probabilities",
            yaxis_title="Probability",
            barmode='group',
            height=350
        )              
        
        st.plotly_chart(fig, use_container_width=True)
        
        if prediction == 1:
            st.error("**High Risk Detected:** Consult a cardiologist immeadiately. Maintain healthy lifestyle, monitor vitals, and consider medication") 
            
        else: 
            st.success("**Low Risk:** Continue healthy habits - exercise regularly, balanced diet, regular checkups")                     
                                            
        with st.expander("Input Summary"):
            st.write(input_data)
            
    # 📌 Sidebar Summary (fixed, non-scrollable)
    with st.sidebar:
        st.header("📖 Model Info")
        st.info(
            """
            **Logistic Regression**
            - Accuracy: ~85%
            - Training: 303 Patients
            - Features: 13 Attributes
            """
        )
        
        st.header("✒️ Model Inputs")
        st.markdown(
            """
            - Chest Pain Type
            - Age & Gender 
            - Blood Pressure
            - Cholestrol Levels
            - Max Heart Rate 
            - Exercise Angina 
            """

        )
        
        st.markdown("---")
        st.caption("This is for educatonal purposes only")
        st.caption("Not a substitute for professional medical advice")
        
        
        
else:
    st.error("Model not found. Run the notebook to generate model files.")








































    
        