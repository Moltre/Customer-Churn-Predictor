import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    preprocessor = joblit.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_model()

st.title("🔮 Customer Churn Predictor")
st.markdown("**Live prediction using XGBoost**")

# Simple inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.slider("Monthly Charges", 18.0, 118.0, 70.0)
total_charges = st.slider("Total Charges", 0.0, 8684.0, 1000.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])

if st.button("🚀 Predict", type="primary"):
    # Input data
    data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly],
        'TotalCharges': [total_charges],
        'Contract': [contract],
        'InternetService': [internet]
    })
    
    # Predict
    X = preprocessor.transform(data)
    prob = model.predict_proba(X)[0, 1]
    
    # Show result
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Risk", f"{prob:.1%}")
    with col2:
        if prob > 0.5:
            st.error("❌ High Risk")
        else:
            st.success("✅ Low Risk")
    
    st.balloons()
