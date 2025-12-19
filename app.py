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

st.title("Heart Disease Predictor")
st.markdown("Predict heart disease risk using Machine Learning")
st.markdown("---")
























    
        